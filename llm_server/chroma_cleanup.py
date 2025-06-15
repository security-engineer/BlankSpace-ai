import chromadb
from dotenv import load_dotenv
import os

def cleanup_chromadb_collection():
    """
    ChromaDB에 연결하여 지정된 컬렉션을 삭제합니다.
    """
    load_dotenv()
    
    CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
    COLLECTION_NAME = "products"

    print("--- ChromaDB 컬렉션 삭제 시작 ---")
    print(f"삭제 대상 컬렉션: '{COLLECTION_NAME}' (서버: {CHROMA_HOST}:{CHROMA_PORT})")

    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        print(f"'{COLLECTION_NAME}' 컬렉션 삭제를 시도합니다...")
        client.delete_collection(name=COLLECTION_NAME)
        print(f"✅ 성공: '{COLLECTION_NAME}' 컬렉션이 성공적으로 삭제되었습니다.")

    except ValueError:
        print(f"⚠️ 정보: '{COLLECTION_NAME}' 컬렉션이 이미 존재하지 않아 삭제할 수 없습니다. 정상적인 상태입니다.")
    except Exception as e:
        print(f"❌ 오류: 컬렉션 삭제 중 예상치 못한 오류 발생: {e}")
    finally:
        print("--- 삭제 작업 종료 ---")

if __name__ == "__main__":
    cleanup_chromadb_collection() 