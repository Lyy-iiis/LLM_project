import os
import requests

# 从.env文件中读取API密钥
api_key = os.getenv("OPENAI_API_KEY")

# 图片和文字查询
image_path = "/ssdshare/style/Phigros-Assets/【曲绘】支线章节三 盗乐行/Quantum Hyperspace.png"
text_query = "describe the image. More details are better. Ignore the text. You may describe in different aspects, such as content, color, style and so on."
# /ssdshare/style/Phigros-Assets/【曲绘】支线章节三 盗乐行/Quantum Hyperspace.png
# 构建API请求
url = "https://api.gpt4o.com/query"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "image": image_path,
    "text": text_query
}

# 发送API请求
response = requests.post(url, headers=headers, json=data)

# 处理API响应
if response.status_code == 200:
    result = response.json()
    # 处理结果
    print(result)
else:
    print("Error:", response.text)