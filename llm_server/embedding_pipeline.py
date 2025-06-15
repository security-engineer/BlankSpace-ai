import os
import pymongo
from dotenv import load_dotenv
from pymongo.database import Database
from pymongo import MongoClient
import chromadb
import traceback
from pymongo.errors import ConnectionFailure, NetworkTimeout, ConfigurationError
import logging
import sys

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection(db_name="blankspace-shopping-mall"):
    """MongoDB에 연결하고 데이터베이스 객체를 반환합니다."""
    try:
        # Docker 내부 통신을 위해 서비스 이름으로 주소를 고정합니다.
        # .env 파일의 영향을 받지 않도록 하여 안정성을 높입니다.
        mongo_uri = "mongodb://165.229.86.157:8080/"
        
        client = MongoClient(mongo_uri, 
                             serverSelectionTimeoutMS=5000, # 5초 안에 서버를 못찾으면 타임아웃
                             connectTimeoutMS=10000) # 10초 안에 연결 못하면 타임아웃
        
        # 연결을 강제로 시도하여 성공/실패를 즉시 확인
        client.admin.command('ping') 
        
        logging.info("MongoDB에 성공적으로 연결되었습니다.")
        return client.get_database(db_name), client
    except ConnectionFailure as e:
        logging.error(f"MongoDB 연결 실패: {e}")
        return None, None
    except Exception as e:
        print(f"MongoDB 연결 중 예기치 않은 오류 발생: {e}")
        return None, None

def fetch_all_products_from_mongodb():
    """MongoDB에서 모든 상품 데이터를 가져와 메모리에 로드하고 연결을 즉시 종료합니다."""
    db, client = get_db_connection()
    if client is None or db is None:
        raise ConnectionFailure("MongoDB 연결에 실패하여 데이터를 가져올 수 없습니다.")

    logging.info("MongoDB에서 모든 상품 데이터를 메모리로 로드합니다...")
    products_collection = db["products"]
    product_list = []
    try:
        # isDeleted가 False이고 status가 'active'인 상품만 가져옵니다.
        product_list = list(products_collection.find({"isDeleted": False, "status": "active"}))
        logging.info(f"성공적으로 {len(product_list)}개의 활성 상품 데이터를 메모리에 로드했습니다.")
        return product_list
    except NetworkTimeout as e:
        logging.error(f"MongoDB 데이터 로딩 중 네트워크 타임아웃 발생: {e}")
        raise
    finally:
        # 데이터 로드 후 즉시 연결 종료
        if client:
            client.close()
            logging.info("MongoDB 연결을 닫았습니다.")

def preprocess_product_data(product: dict) -> tuple[str, dict, str]:
    """MongoDB의 상품 문서를 전처리하여 임베딩할 텍스트와 메타데이터를 생성합니다."""
    # 필수 필드 추출
    doc_id = str(product['_id'])
    name = product.get('name', '이름 없음')
    description = product.get('description', '설명 없음')
    price = product.get('price', 0)

    # 카테고리 정보 처리
    category = product.get('category', [])
    category_str = ", ".join(category) if isinstance(category, list) else str(category)

    # 재고 정보 처리
    stock = product.get('stock', {})
    stock_str = "재고 정보 없음"
    if stock and isinstance(stock, dict):
        valid_stock = {size: qty for size, qty in stock.items() if isinstance(qty, int)}
        if valid_stock:
            stock_str = ", ".join([f"{size}: {qty}개" for size, qty in valid_stock.items()])

    # --- 임베딩될 텍스트 내용 구성 (height, weight 제외) ---
    content_parts = [
        f"상품명: {name}",
        f"카테고리: {category_str}",
        f"가격: {price}원",
        f"상품 상세설명: {description}",
        f"사이즈별 재고: {stock_str}"
    ]

    # 세탁 방법 정보 추가
    wash_methods = product.get('washMethods', [])
    if wash_methods:
        if isinstance(wash_methods, list) and all(isinstance(i, dict) for i in wash_methods):
            methods = [item.get('method', '') for item in wash_methods if item.get('method')]
            wash_str = ", ".join(methods)
        else:
            wash_str = ", ".join(wash_methods)
        if wash_str:
            content_parts.append(f"세탁 정보: {wash_str}")

    content = "\\n".join(content_parts).strip()
    
    # 이미지 URL 추출
    image_field = product.get('image')
    image_url = ''
    if isinstance(image_field, dict):
        image_url = image_field.get('url', '')
    elif isinstance(image_field, list) and image_field:
        first_item = image_field[0]
        if isinstance(first_item, dict):
            image_url = first_item.get('url', '')
        elif isinstance(first_item, str):
            image_url = first_item

    # --- 최종 메타데이터 (height, weight 제외) ---
    metadata = {
        'product_id': doc_id,
        'sku': product.get('sku', ''),
        'name': name,
        'price': price,
        'category': category_str,
        'description': description,
        'image_url': image_url,
        'stock': stock_str,
    }

    return content, metadata, doc_id

def create_documents_from_products(products: list):
    """상품 데이터를 전처리하여 LangChain 문서 객체 리스트로 변환합니다."""
    documents = []
    for product in products:
        try:
            content, metadata, doc_id = preprocess_product_data(product)
            # content와 metadata를 함께 튜플로 추가하지 않고, Document 객체로 생성합니다.
            # 하지만 ChromaDB의 add_documents는 Document 객체를 받으므로, 여기서 Document를 만들 필요는 없습니다.
            # rag_module 내부에서 처리하거나, add_documents에 맞게 텍스트와 메타데이터를 전달해야 합니다.
            # 여기서는 우선 텍스트와 메타데이터를 분리하여 준비합니다.
            documents.append({"content": content, "metadata": metadata, "id": doc_id})
        except Exception as e:
            product_id = product.get('_id', 'N/A')
            logging.error(f"오류: 상품 ID '{product_id}' 처리 중 문제 발생. 건너뜁니다. 에러: {e}")
            traceback.print_exc()
            continue
    return documents

def main():
    """메인 임베딩 파이프라인 함수"""
    logging.info("상품 데이터 임베딩 파이프라인을 시작합니다.")
    
    product_data = None
    try:
        # 1. I/O 작업: MongoDB에서 데이터 가져오기 (AI 모델 로딩 전)
        product_data = fetch_all_products_from_mongodb()
    except Exception as e:
        logging.error(f"MongoDB에서 데이터를 가져오는 중 심각한 오류 발생: {e}")
        sys.exit(1) # 데이터 없이는 진행 불가

    if not product_data:
        logging.warning("MongoDB에 상품 데이터가 없습니다. 파이프라인을 종료합니다.")
        return

    try:
        # --- 지연 로딩 ---
        # 2. CPU/GPU 집약적 작업: AI 모델 로딩 및 임베딩
        # MongoDB 작업이 모두 끝난 후에 필요한 모듈을 임포트합니다.
        logging.info("필요한 AI 모듈을 로딩합니다...")
        from rag_module import get_rag_module
        from langchain_community.vectorstores import Chroma
        
        logging.info("RAG 모듈을 초기화합니다.")
        rag_module = get_rag_module()
        
        # 3. 문서 생성
        logging.info("LangChain 문서 객체를 생성합니다...")
        documents_to_add = create_documents_from_products(product_data)
        if not documents_to_add:
            logging.warning("생성된 문서가 없습니다. 파이프라인을 종료합니다.")
            return

        # 4. ChromaDB에 데이터 저장
        logging.info(f"생성된 {len(documents_to_add)}개의 문서를 ChromaDB에 저장합니다...")
        
        # add_documents에 맞게 데이터 형식 변환
        contents = [doc["content"] for doc in documents_to_add]
        metadatas = [doc["metadata"] for doc in documents_to_add]
        ids = [doc["id"] for doc in documents_to_add]

        # 멱등성 보장을 위해 기존 컬렉션 삭제
        try:
            logging.info(f"기존 '{rag_module.collection_name}' 컬렉션을 삭제합니다.")
            rag_module.chroma_client.delete_collection(name=rag_module.collection_name)
        except Exception as e:
            logging.warning(f"컬렉션 삭제 중 오류 발생 (아직 존재하지 않을 수 있음, 무시하고 계속 진행): {e}")

        # Chroma 클래스 초기화 시 컬렉션이 없으면 자동으로 생성됨
        vector_store = Chroma(
            client=rag_module.chroma_client,
            collection_name=rag_module.collection_name,
            embedding_function=rag_module.embedding_function,
        )
        
        vector_store.add_texts(texts=contents, metadatas=metadatas, ids=ids)

        logging.info("모든 데이터가 ChromaDB에 성공적으로 저장되었습니다.")
        
    except Exception as e:
        logging.error(f"임베딩 및 저장 과정에서 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 