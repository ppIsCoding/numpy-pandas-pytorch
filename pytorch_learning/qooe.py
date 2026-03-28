from google import genai

client = genai.Client(api_key="")

try:
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="你好，请用中文回答：今天天气怎么样？"
    )
    print(response.text)
except Exception as e:
    print("[错误]:", str(e))
