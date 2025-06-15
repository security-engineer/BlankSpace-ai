import chromadb
from dotenv import load_dotenv
import os

def verify_chromadb_data():
    """
    ChromaDB에 연결하여 'products' 컬렉션의 상태를 확인하고
    저장된 데이터 샘플을 출력합니다.
    """
    load_dotenv()
    
    CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
    CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")

    print("--- ChromaDB 데이터 검증 시작 ---")
    print(f"연결 대상: {CHROMA_HOST}:{CHROMA_PORT}")

    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        # 연결 테스트 (실제로는 list_collections() 등이 호출될 때 연결이 발생)
        collections = client.list_collections()
        print("✅ ChromaDB 연결 성공!")
        
        collection_name = "products"
        print(f"'{collection_name}' 컬렉션 확인 중...")

        # 컬렉션이 존재하는지 확인
        if not any(c.name == collection_name for c in collections):
            print(f"❌ 오류: '{collection_name}' 컬렉션이 존재하지 않습니다.")
            return

        collection = client.get_collection(name=collection_name)
        
        # 1. 총 문서 개수 확인
        count = collection.count()
        print(f"✅ 총 문서 개수: {count}개")

        if count > 0:
            # 2. 샘플 데이터 확인
            print("\n--- 샘플 데이터 (최대 5개) ---")
            sample_data = collection.get(
                limit=58,
                include=["metadatas", "documents"] # embeddings는 너무 길어서 제외
            )
            
            ids = sample_data['ids']
            metadatas = sample_data['metadatas']
            documents = sample_data['documents']

            for i in range(len(ids)):
                print(f"\n[문서 ID: {ids[i]}]")
                print(f"  - 메타데이터: {metadatas[i]}")
                # print(f"  - 내용: {documents[i][:200]}...") # 내용이 길 경우 일부만 출력
                print("-" * 20)

            print("\n✅ 검증 완료: 데이터가 성공적으로 저장된 것으로 보입니다.")

    except Exception as e:
        print(f"❌ 검증 중 오류 발생: {e}")
    finally:
        print("--- 검증 종료 ---")

if __name__ == "__main__":
    verify_chromadb_data() 