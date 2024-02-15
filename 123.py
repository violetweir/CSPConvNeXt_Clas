import openai
openai.api_key = 'dummy'
openai.api_base = 'http://222.31.135.54:8978'
result = openai.Completion.create(
    engine='codegen', prompt='def hello', max_tokens=16, temperature=0.1)
print(result)
'''
<OpenAIObject text_completion id=cmpl-dmhoeHmcw9DJ4NeqOJDQVKv3iivJ0 at 0x7fe7a81d42c0> JSON: {
  "id": "cmpl-dmhoeHmcw9DJ4NeqOJDQVKv3iivJ0",
  "choices": [
    {
      "text": "_world():\n    print(\"Hello World!\")\n\n\n#",
      "index": 0,
      "finish_reason": "stop",
      "logprobs": null,
    }
  ],
  "usage": {
    "completion_tokens": null,
    "prompt_tokens": null,
    "total_tokens": null
  }
}
'''