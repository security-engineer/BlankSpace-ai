import chromadb
from pprint import pprint

# ChromaDB 서버에 연결 (docker-compose에서 포트를 8000으로 열어두었기 때문)
client = chromadb.HttpClient(host="localhost", port=8000)

print("✅ ChromaDB 서버에 성공적으로 연결되었습니다.\n")

# 1. 모든 컬렉션(테이블 같은 개념) 목록 보기
try:
    collections = client.list_collections()
    print("📚 현재 존재하는 컬렉션 목록:")
    if not collections:
        print("- (아직 컬렉션이 없습니다)")
    for collection in collections:
        print(f"- {collection.name} (데이터 {collection.count()}개)")
    print("-" * 30)

    # 2. 'product_test' 컬렉션의 모든 데이터 조회하기
    if any(c.name == 'product_test' for c in collections):
        print("🔍 'product_test' 컬렉션 데이터 조회:")
        test_collection = client.get_collection(name="product_test")
        
        # .get() 메서드에 아무 인자도 주지 않으면 모든 데이터를 가져옵니다.
        all_data = test_collection.get()
        
        pprint(all_data)
    else:
        print("🔍 'product_test' 컬렉션이 존재하지 않습니다.")

except Exception as e:
    print(f"❌ 데이터를 조회하는 중 오류 발생: {e}")
