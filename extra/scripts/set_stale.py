import asyncio
from argparse import ArgumentParser

from aiohttp import ClientSession


async def list_workers(session: ClientSession, task: str) -> list[int]:
    response = await session.get('list_workers')
    response.raise_for_status()
    data = await response.json()
    return [worker['id'] for worker in data.get(task, {}).get('workers', {}).values()]


async def set_stale(session: ClientSession, task: str, worker: int, is_stale: bool) -> None:
    response = await session.post('mark_stale', json={
        'name': task,
        'worker_id': worker,
        'stale': is_stale
    })
    response.raise_for_status()
    print(f'Worker {task}#{worker} set stale={is_stale}')


async def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--controller', default='http://localhost:5020/api')
    parser.add_argument('action', choices=['stale', 'unstale'])
    parser.add_argument('task')
    parser.add_argument('from_id', type=int)
    parser.add_argument('to_id', type=int)
    args = parser.parse_args()

    if args.controller[-1] != '/':
        args.controller += '/'
    async with ClientSession(args.controller) as session:
        workers = await list_workers(session, args.task)
        if not workers:
            print(f'No workers found for task "{args.task}"')
            return

        futures = []
        for worker_id in workers:
            if args.from_id <= worker_id <= args.to_id:
                is_stale = (args.action == 'stale')
                futures.append(set_stale(session, args.task, worker_id, is_stale))
        await asyncio.gather(*futures)


if __name__ == '__main__':
    asyncio.run(main())
