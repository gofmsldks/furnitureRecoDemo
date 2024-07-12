from dotenv import load_dotenv
import os
# API KEY 정보로드
load_dotenv()
print(f"[API KEY]\n{os.environ['OPENAI_API_KEY']}")




# 색상 데이터베이스 설정 (예시)
color_database = {}
with open('./color_data.txt', 'r') as file:
    for line in file:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            color_database[key] = value
print(color_database)