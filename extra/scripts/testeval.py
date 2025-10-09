import asyncio
import os
import sys
import traceback
from argparse import ArgumentParser
from os.path import dirname, join
from typing import List, Tuple, Union

import pandas as pd
from aiohttp import ClientSession
from openai import OpenAI

client = OpenAI()


def print_message(message: Union[List[dict], dict]):
    if isinstance(message, list):
        for item in message:
            print_message(item)
        return

    role = message["role"].upper()
    if message.get('tool_call_id'):
        role += f' {message["tool_call_id"]}'

    content = message.get('content', [])
    if isinstance(content, str):
        text = content.replace('\r', r'\r').replace('\n', r'\n')
        if text:
            print(f'[{role}] {text}', flush=True)
    else:
        for item in content:
            if item.get('type') == 'text':
                text = item['text'].replace('\r', r'\r').replace("\n", r"\n")
                if text:
                    print(f'[{role}] {text}', flush=True)
            else:
                print(f'[{role}] ({item["type"]}) omitted', flush=True)

    for tool_call in message.get('tool_calls', []) or []:
        print(f'[{role}] tool call {tool_call}', flush=True)


def remove_earliest_messages(messages: List[dict]) -> List[dict]:
    if not messages:
        return messages

    protected_indices = set()
    protected_indices.add(0)
    protected_indices.add(1)
    for i, m in list(enumerate(messages))[::-1]:
        if m['role'] == 'tool':
            protected_indices.add(i)
        else:
            break

    assistant_index = next(
        (i for i, m in enumerate(messages)
         if m['role'] == 'assistant' and i not in protected_indices),
        None
    )
    if assistant_index is None:
        return messages  # nothing to remove

    assistant_msg = messages[assistant_index]
    tool_call_ids = set()
    if 'tool_calls' in assistant_msg:
        tool_call_ids.update(call['id'] for call in assistant_msg['tool_calls'])

    indices_to_remove = set()
    indices_to_remove.add(assistant_index)
    for i in range(assistant_index + 1, len(messages)):
        msg = messages[i]
        if msg['role'] == 'tool' and msg.get('tool_call_id') in tool_call_ids:
            if i not in protected_indices:
                indices_to_remove.add(i)
        else:
            break

    modified_messages = []
    for i, msg in enumerate(messages):
        if i in indices_to_remove:
            msg['removed'] = True
        modified_messages.append(msg)
    return modified_messages


async def worker_start_sample_only(args, session: ClientSession, task_index: str) -> bool:
    try:
        response = await session.post(f'{args.controller}/start_sample', json={
            'name': args.task_name[0],
            'index': task_index
        })
        response.raise_for_status()
        data = await response.json()
        assert data['messages'], 'No messages returned from start_sample'
        if args.verbose:
            print(f'Started sample for task {task_index}')
        await session.post(f'{args.controller}/cancel', headers={
            'session_id': response.headers['session_id']
        })
        await asyncio.sleep(20)  # wait for task cancellation to avoid stressing the workers
        return True
    except Exception as e:
        print(f'Failed to start sample for task {task_index}: {e}', file=sys.stderr)
        return False


async def worker(args, session: ClientSession, task_index: str) -> Tuple[List[dict], Union[int, Exception]]:
    while True:
        try:
            response = await session.post(f'{args.controller}/start_sample', json={
                'name': args.task_name[0],
                'index': task_index
            })
            response.raise_for_status()
            session_id = response.headers['session_id']
            break
        except:
            if args.verbose:
                print(f'Failed to start sample for task {task_index}, retrying later...', file=sys.stderr)
                traceback.print_exc()
            await asyncio.sleep(10)
    messages = []
    try:
        data = await response.json()
        messages = data['messages']
        tools = data['tools']

        if args.verbose:
            print(f'session_id: {session_id}')
            print_message(data['messages'])

        while True:
            for tries in range(10):
                try:
                    response = await asyncio.to_thread(
                        client.chat.completions.create,
                        model=args.model,
                        messages=[i for i in messages if 'removed' not in i],
                        tools=tools
                    )
                    message = response.choices[0].message.model_dump()
                    break
                except Exception as e:
                    print(e)
                    if tries == 4:
                        raise
            messages.append(message)

            if args.verbose:
                print_message(message)

            response = await session.post(f'{args.controller}/interact', headers={
                'session_id': session_id
            }, json={
                'messages': [message]
            })
            response.raise_for_status()
            data = await response.json()

            if args.verbose:
                print_message(data['messages'])

            if data['finish']:
                return messages, data['reward']
            elif args.cover:
                messages = data['messages']
            else:
                messages.extend(data['messages'])
    except Exception as e:
        try:
            await session.post(f'{args.controller}/cancel', headers={
                'session_id': session_id
            })
        except:
            traceback.print_exc()
        return messages, e


async def run_tasks(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = join(args.output_dir, f'{args.task_name[0]}-eval.jsonl')

    # connect to controller
    session = ClientSession()
    response = await session.post(f'{args.controller}/sync_all')
    response.raise_for_status()

    # get task indices
    if args.index:
        task_indices = [args.index]
        args.runs = 1
    else:
        response = await session.get(f'{args.controller}/get_indices', params={
            'name': args.task_name[0]
        })
        response.raise_for_status()
        task_indices = [
            i for i in await response.json()
            if i != -1
        ]

    # create output file
    df_lock = asyncio.Lock()
    if os.path.exists(output_file):
        df = pd.read_json(output_file, lines=True)
    else:
        df = pd.DataFrame(columns=['model', 'task_index', 'run_number', 'result', 'messages', 'timestamp'])

    # count completed runs
    completed_runs = set(
        (row['task_index'], int(row['run_number']))
        for _, row in df.iterrows()
        if row['model'] == args.model and row['result'] != 'error'
    )

    # create tasks
    semaphore = asyncio.Semaphore(args.concurrency)
    async def sem_task(task_index: str, run_number: int):
        async with semaphore:
            if args.start_sample:
                await worker_start_sample_only(args, session, task_index)
                return

            if args.verbose:
                print(f'\n========== task {task_index} run {run_number} ==========\n', flush=True)

            try:
                messages, result = await worker(args, session, task_index)
            except Exception as e:
                messages = []
                result = e

            if isinstance(result, Exception):
                print(f'Error in task {task_index} run {run_number}:', file=sys.stderr, flush=True)
                traceback.print_exception(result)
                result = 'error'
            else:
                print(f'Completed task {task_index} run {run_number} with result {result}', flush=True)

            if result == 'error' and len(messages) == 0:
                return

            async with df_lock:
                index = df[(df['model'] == args.model) & (df['task_index'] == task_index) & (df['run_number'] == run_number)].index
                if len(index) > 0:
                    df.at[index[0], 'result'] = result
                    df.at[index[0], 'messages'] = messages
                    df.at[index[0], 'timestamp'] = pd.Timestamp.utcnow()
                else:
                    df.loc[len(df)] = [args.model, task_index, run_number, result, messages, pd.Timestamp.utcnow()]
                df.sort_values(by=['model', 'task_index', 'run_number'], inplace=True)
                df.to_json(output_file, index=False, orient='records', lines=True)

    coroutines = []
    for run_number in range(1, args.runs + 1):
        for task_index in task_indices:
            if (task_index, run_number) in completed_runs:
                continue
            coroutines.append(sem_task(task_index, run_number))
    await asyncio.gather(*coroutines, return_exceptions=False)

    await session.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('-m', '--model', type=str, required=True, help='name of the model to be evaulated')
    arg_parser.add_argument('-c', '--controller', type=str, default='http://localhost:5020/api', help='controller URL')
    arg_parser.add_argument('-j', '--concurrency', type=int, default=64, help='number of concurrent tasks to run')
    arg_parser.add_argument('-n', '--runs', type=int, default=5, help='number of runs for each task')
    arg_parser.add_argument('-o', '--output-dir', type=str, default=join(dirname(dirname(__file__)), 'results'), help='directory to store the results')
    arg_parser.add_argument('-v', '--verbose', action='store_true', help='print more information')
    arg_parser.add_argument('-i', '--index', type=str, required=False, help='if specified, only test the task of given index; invalidates --runs')
    arg_parser.add_argument('--start-sample', action='store_true', help='start sample only to test if the task can be started')
    arg_parser.add_argument('--cover', action='store_true', help='cover messages in each interaction')
    arg_parser.add_argument('task_name', type=str, nargs=1, help='name of the task to be evaluated')

    args = arg_parser.parse_args()
    if args.verbose and args.concurrency > 1:
        print('Error: verbose mode is not compatible with concurrent evaluation', file=sys.stderr)
        sys.exit(1)
    if args.start_sample:
        if args.runs > 1:
            print('Warning: --start-sample is incompatible with --runs, setting runs to 1', file=sys.stderr)
            args.runs = 1
        if args.index:
            print('Warning: --start-sample is incompatible with --index, ignoring --index', file=sys.stderr)
            args.index = None

    asyncio.run(run_tasks(args))
