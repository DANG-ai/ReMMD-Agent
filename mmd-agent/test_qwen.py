from openai import OpenAI

# 请将{{your api key}}替换为你的API密钥
api_key = 'qwen'

if __name__ == '__main__':
    client = OpenAI(
        base_url="http://YOUR_QWEN_27B_ENDPOINT/v1",
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model="qwen3.6-27b",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant."
        },
            {"role": "user",
             "content": "Give me a python code to print 'Hello, World!'"}]
    )
    print(completion.choices[0].message.content)