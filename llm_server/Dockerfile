# 공식 Python 런타임을 부모 이미지로 사용합니다.
FROM python:3.10-slim

# 컨테이너의 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 시스템 의존성 패키지를 설치합니다 (whisper에 필요한 ffmpeg 등).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && apt-get upgrade \
    && apt update -y && apt upgrade -y \
    && apt install -y procps vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 파일을 컨테이너의 /app 디렉토리로 복사합니다.
COPY requirements.txt .

# requirements.txt에 명시된 파이썬 패키지들을 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# 모델 다운로드를 위한 캐시 폴더를 명시적으로 지정하고 생성
ENV HUGGINGFACE_HUB_HOME /app/huggingface_cache
RUN mkdir -p /app/huggingface_cache

# 애플리케이션 코드는 docker-compose.yml의 볼륨 마운트를 통해 제공됩니다.
# 볼륨 마운트: ./llm_server:/app

# 애플리케이션 실행 명령어는 docker-compose.yml에서 오버라이드하여 사용합니다. 