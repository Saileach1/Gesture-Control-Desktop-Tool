import requests
import os

# 下载模型文件
url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
output_path = "models/hand_landmarker.task"

print(f"正在下载模型文件...")
print(f"URL: {url}")
print(f"保存路径: {output_path}")

try:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 写入文件
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"模型文件下载成功！")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

except Exception as e:
    print(f"下载失败: {e}")