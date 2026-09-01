"""Minimal client for DigitalOcean Serverless Inference.

Usage:

from do_client import DOClient

client = DOClient()
result = client.chat("Why is the sky blue?", model="deepseek-v4-pro")

print(result["choices"][0]["message"]["content"])
print(result["usage"]["total_tokens"])

"""

import os

import requests


class DOClient:
    URL = "https://inference.do-ai.run/v1/chat/completions"

    def __init__(self, api_key=None, max_tokens=3125):
        self.api_key = api_key or os.environ["DO_KEY"]
        self.max_tokens = max_tokens
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        })

    def chat(self, prompt, model, max_tokens=None):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.max_tokens,
        }
        r = self.session.post(self.URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()