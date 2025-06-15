import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import traceback
import whisper
from fastapi.responses import JSONResponse, StreamingResponse
import tempfile
import io
from TTS.api import TTS
from pathlib import Path

# conversation_manager 모듈이 로드될 때 모든 초기화가 발생합니다.
from conversation_manager import conversation_manager_instance

# 환경 변수 로드 (.env 파일이 있다면)
load_dotenv()

app = FastAPI(
    title="BlankSpace AI Assistant API",
    description="대화형 상품 추천 AI 비서 API입니다.",
    version="1.0.0"
)

# CORS 설정 - 모든 출처 허용으로 변경
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 모델 로딩 ---

# STT 모델 (Whisper)
try:
    stt_model = whisper.load_model("base")
    print("Whisper STT 모델 'base' 로드 완료.")
except Exception as e:
    print(f"Whisper STT 모델 로드 실패: {e}")
    stt_model = None

# TTS 모델 (Coqui TTS)
def load_tts_model():
    """Coqui TTS 모델을 로드합니다."""
    try:
        # 다국어 모델을 사용하고, 언어는 'ko'로 지정합니다.
        # 기존 모델이 문제가 있을 경우 대체 모델을 시도합니다
        try:
            model_name = "tts_models/multilingual/multi-dataset/your_tts"
            print(f"Coqui TTS 다국어 모델 로딩 시작: {model_name}")
            # CPU 모드로 먼저 시도합니다
            tts_instance = TTS(model_name, gpu=False) 
            print("Coqui TTS 모델 로드 완료 (CPU 모드).")
            return tts_instance
        except Exception as e:
            print(f"your_tts 모델 로드 실패: {e}")
            print("대체 TTS 모델을 시도합니다...")
            
            # 대체 모델 시도 - 한국어 전용 모델
            model_name = "tts_models/ko/glowtts/kokoro"
            print(f"대체 TTS 모델 로딩 시작: {model_name}")
            tts_instance = TTS(model_name, gpu=False)
            print("대체 Coqui TTS 모델 로드 완료 (CPU 모드).")
            return tts_instance
    except Exception as e:
        print(f"모든 TTS 모델 로드 실패: {e}")
        traceback.print_exc()
        print("TTS 기능 없이 서버를 계속 실행합니다.")
        return None

tts_model = load_tts_model()

# --- API 요청/응답 모델 ---

class TextChatRequest(BaseModel):
    sessionId: str
    message: str
    pageInfo: Dict[str, Any]

class Action(BaseModel):
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)

class TextChatResponse(BaseModel):
    message: str
    action: Optional[Action] = None

class ClearHistoryRequest(BaseModel):
    sessionId: str

class TTSRequest(BaseModel):
    text: str

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    # conversation_manager_instance가 제대로 로드되었는지 확인
    if conversation_manager_instance is None:
        print("경고: ConversationManager 초기화 실패. 챗봇 기능이 동작하지 않을 수 있습니다.")

@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok"}

@app.post("/text-chat", response_model=TextChatResponse, summary="AI와 텍스트 대화")
async def text_chat(request: TextChatRequest):
    if conversation_manager_instance is None:
        # RAG 모듈이 초기화되지 않았을 때 더미 응답 반환
        print("경고: ConversationManager가 초기화되지 않아 더미 응답을 반환합니다.")
        return TextChatResponse(
            message="죄송합니다. 현재 AI 서비스가 초기화 중입니다. 잠시 후 다시 시도해주세요.",
            action=None
        )
    try:
        response_data = conversation_manager_instance.handle_conversation_turn(
            request.sessionId, request.message, request.pageInfo
        )
        return TextChatResponse(message=response_data["message"], action=response_data.get("action"))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history", status_code=204, summary="대화 기록 초기화")
async def clear_history(request: ClearHistoryRequest):
    if conversation_manager_instance is None:
        raise HTTPException(status_code=503, detail="AI 서비스가 현재 사용 불가능합니다.")
    try:
        conversation_manager_instance.clear_session(request.sessionId)
    except Exception as e:
        raise HTTPException(status_code=500, detail="대화 기록 삭제 중 오류 발생")

@app.post("/stt", summary="음성-텍스트 변환 (STT)")
async def speech_to_text(audio: UploadFile = File(...)):
    if stt_model is None:
        raise HTTPException(status_code=503, detail="STT 서비스 사용 불가.")
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as temp_audio_file:
            content = await audio.read()
            temp_audio_file.write(content)
            temp_audio_file.flush()
            result = stt_model.transcribe(temp_audio_file.name, fp16=False)
            return JSONResponse(content={"text": result.get("text", "")})
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="음성 처리 중 오류 발생")

@app.post("/tts", summary="텍스트-음성 변환 (TTS)")
async def text_to_speech(request: TTSRequest):
    if tts_model is None:
        # TTS 모델 로드 실패 시 오류 메시지 반환
        return JSONResponse(
            status_code=503,
            content={
                "error": "TTS 서비스를 사용할 수 없습니다.",
                "message": "TTS 모델 로드에 실패했습니다. 관리자에게 문의하세요."
            }
        )
    try:
        print(f"TTS 요청 수신: text='{request.text}'")
        
        # 입력 텍스트가 비어있는지 확인
        if not request.text or len(request.text.strip()) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "텍스트가 비어있습니다.",
                    "message": "TTS 변환을 위한 텍스트를 입력해주세요."
                }
            )
            
        try:
            print("TTS 모델 호출 시작...")
            
            # 모델 유형에 따라 다른 파라미터 사용
            if "your_tts" in tts_model.model_name:
                # your_tts 모델은 language와 speaker 파라미터가 필요
                wav_chunks = tts_model.tts(text=request.text, speaker=tts_model.speakers[0], language='ko')
            else:
                # 다른 모델은 language 파라미터만 사용 (가능한 경우)
                try:
                    wav_chunks = tts_model.tts(text=request.text, language='ko')
                except TypeError:
                    # language 파라미터를 지원하지 않는 경우
                    wav_chunks = tts_model.tts(text=request.text)
                    
            print(f"TTS 모델 호출 완료. wav_chunks 타입: {type(wav_chunks)}, 길이: {len(wav_chunks) if isinstance(wav_chunks, list) else 'N/A'}")

            # 리스트 형태의 오디오 데이터를 하나의 byte-stream으로 변환합니다.
            # FastAPI의 StreamingResponse에 맞게 generator를 생성합니다.
            def audio_stream_generator():
                try:
                    print("오디오 스트림 제너레이터 시작.")
                    # Coqui TTS는 wav 헤더를 포함하지 않은 raw PCM 데이터를 반환할 수 있으므로,
                    # 완전한 WAV 파일을 만들기 위해 헤더를 직접 생성하고 데이터를 붙여줍니다.
                    import wave
                    import numpy as np

                    print("오디오 데이터 처리 시작...")
                    # 데이터를 numpy 배열로 변환
                    audio_data = np.array(wav_chunks, dtype=np.int16)
                    print(f"Numpy 배열 변환 완료. 데이터 shape: {audio_data.shape}")

                    # BytesIO 객체에 WAV 데이터 쓰기
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, 'wb') as wf:
                        wf.setnchannels(1)  # Mono
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(tts_model.synthesizer.output_sample_rate)
                        wf.writeframes(audio_data.tobytes())
                    
                    wav_io.seek(0)
                    print("WAV 데이터 생성 완료, 스트리밍 시작.")
                    yield wav_io.read()
                    print("스트리밍 완료.")
                except Exception as e:
                    print("!!! 오디오 스트림 제너레이터 내부에서 오류 발생 !!!")
                    traceback.print_exc()

            return StreamingResponse(audio_stream_generator(), media_type="audio/wav")
        except Exception as e:
            print("!!! TTS 엔드포인트 메인 로직에서 오류 발생 !!!")
            traceback.print_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "error": "음성 생성 중 오류 발생",
                    "message": str(e)
                }
            )
    except Exception as e:
        print("!!! TTS 엔드포인트 메인 로직에서 오류 발생 !!!")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": "음성 생성 중 오류 발생",
                "message": str(e)
            }
        )

# 개발용 서버 실행 (docker-compose에서 사용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 