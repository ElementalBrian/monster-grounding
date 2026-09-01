#"kimi-k2.5", "qwen3.8-max", "deepseek-v4-pro", "kimi-k2.6", "glm-5.2", "kimi-k3", "qwen3.5-397b-a17b", "deepseek-4-flash"

import os

import requests

api_key = os.environ["DO_KEY"]

url = "https://inference.do-ai.run/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

model_list = ["kimi-k2.6"]

content = "test test. hello? are you a monster rancher?"

# model = "kimi-k2.6"
for model in model_list:
    data = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": f"{content}"
            }
        ],
        "max_tokens": 10000
    }
    print("------------------------------------------------------------------------------------------------------")
    print(model)
    print(data['messages'][0]['content'])
    print()

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    usage = response.json()["usage"]

    print(f"prompt:     {usage['prompt_tokens']}")
    print(f"completion: {usage['completion_tokens']}")
    print(f"total:      {usage['total_tokens']}")
    print()
    print(response.json()["choices"][0]["message"]["content"])