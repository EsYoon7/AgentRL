from httpx import Client
from openai import OpenAI

# create http client and openai client
client = Client(base_url='http://localhost:5020/api')
openai = OpenAI()

# load indices
indices = client.get('/get_indices', params={
    'name': 'simple-calculator'
}).json()
print(f'Loaded {len(indices)} indices to evaluate')

# evaluate each index (1 concurrency)
results = []
for index in indices:
    print(f'Evaluating task index {index}')

    # we need to maintain the full conversation history
    messages = []

    # start a sample
    response = client.post('/start_sample', json={
        'name': 'simple-calculator',
        'index': index
    })

    # a session id will be returned by `start_sample`,
    # use it to `interact` with the environment.
    session_id = response.headers.get('session_id')

    # add system and user prompts to the message history
    messages.extend(response.json()['messages'])

    # save tool definitions for the agent to call
    tools = response.json()['tools']

    # the interaction loop
    while True:

        # request openai api to get the next message
        agent_response = openai.chat.completions.create(model='gpt-5-nano-2025-08-07', messages=messages, tools=tools)
        agent_message = agent_response.choices[0].message.model_dump(mode='json')

        # add the new message to the history
        messages.append(agent_message)

        # send the message to the environment
        env_response = client.post('/interact', headers={
            'session_id': session_id
        }, json={
            'messages': [agent_message]
        }).json()

        # check if the task is done
        if env_response['finish']:
            print(f'Task index {index} done with status {env_response["status"]} and reward {env_response["reward"]}')
            results.append(env_response['reward'])
            break

        # save the new messages from the environment to the history
        messages.extend(env_response['messages'])

# calculate overall accuracy
accuracy = sum(results) / len(results)
print(f'Overall accuracy: {accuracy}')
