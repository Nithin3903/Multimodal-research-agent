from ollama import chat


image_path = "data/test.png"

response = chat(
    model="qwen2.5vl:7b",
    messages=[
        {
            "role": "user",
            "content": "Analyze this image carefully. Describe what you see and explain the important information.",
            "images": [image_path],
        }
    ],
)

print("\nAI RESPONSE:\n")
print(response.message.content)