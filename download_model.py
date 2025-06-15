from sentence_transformers import SentenceTransformer
import os

# 다운로드할 폴더를 명시적으로 지정
cache_dir = "huggingface_cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

model_name = 'paraphrase-multilingual-mpnet-base-v2'
print(f"'{model_name}' 모델 다운로드를 시작합니다...")
print(f"저장 위치: {os.path.abspath(cache_dir)}")

# cache_folder를 지정하여 모델을 다운로드
model = SentenceTransformer(model_name, cache_folder=cache_dir)

print("모델 다운로드가 성공적으로 완료되었습니다!") 