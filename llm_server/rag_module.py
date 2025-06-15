import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, models
import os
import time
from dotenv import load_dotenv
from typing import Any, Dict, List


class SafeSentenceTransformerEmbeddings(BaseModel, Embeddings):
    """
    LangChain과 호환되면서 safetensors를 안전하게 로드하는 커스텀 임베딩 클래스.
    sentence-transformers 라이브러리를 직접 사용하여 모델을 올바르게 인스턴스화합니다.
    """
    client: Any = Field(default=None, exclude=True)  # Pydantic 유효성 검사에서 제외
    model_name: str = "jhgan/ko-sbert-nli"
    cache_folder: str = None
    model_kwargs: Dict[str, Any] = {}
    encode_kwargs: Dict[str, Any] = {}

    model_config = {
        "protected_namespaces": (),
    }

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic V2의 공식 초기화 후크.
        필드 유효성 검사 후에 SentenceTransformer 클라이언트를 생성합니다.
        """
        super().model_post_init(__context)
        try:
            # sentence-transformers의 모듈을 수동으로 구성하여 모델을 생성
            word_embedding_model = models.Transformer(
                model_name_or_path=self.model_name,
                model_args={'use_safetensors': True},  # safetensors 사용 강제
                cache_dir=self.cache_folder
            )
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            self.client = SentenceTransformer(
                modules=[word_embedding_model, pooling_model]
            )
        except Exception as e:
            raise RuntimeError(f"SentenceTransformer 모델 로딩 실패: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """주어진 텍스트 목록에 대한 임베딩을 계산합니다."""
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        embeddings = self.client.encode(texts, **self.encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """단일 텍스트에 대한 임베딩을 계산합니다."""
        text = text.replace("\n", " ")
        embedding = self.client.encode(text, **self.encode_kwargs)
        return embedding.tolist()


class RAGModule:
    """
    RAG (Retrieval-Augmented Generation) 관련 기능을 관리하는 클래스.
    - SentenceTransformer를 이용한 텍스트 임베딩
    - ChromaDB를 이용한 벡터 검색
    """
    def __init__(self, collection_name="products"):
        """
        RAGModule을 초기화합니다.
        
        Args:
            collection_name (str): 사용할 ChromaDB 컬렉션 이름. 기본값은 "products".
        """
        load_dotenv()
        
        try:
            # 1. 새로 정의한 안전한 임베딩 클래스 사용
            model_name = "jhgan/ko-sbert-nli"
            cache_folder = os.path.join(os.path.dirname(__file__), '..', 'huggingface_cache')
            print(f"[RAG Module] SentenceTransformer 임베딩 모델 로드를 시작합니다...")
            print(f"[RAG Module] 사용하는 모델: {model_name}, 캐시 폴더: {cache_folder}")
            
            self.embedding_function = SafeSentenceTransformerEmbeddings(
                model_name=model_name,
                cache_folder=cache_folder
            )
            print("[RAG Module] 임베딩 모델 로드 완료.")

            # 2. ChromaDB 연결
            chroma_host = os.getenv("CHROMA_HOST", "chromadb")
            print(f"ChromaDB 연결 시도 중: {chroma_host}:8000")
            
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host, 
                port=8000,
            )
            self.chroma_client.heartbeat() # 연결 테스트
            print("ChromaDB 연결 성공.")
            
            # 3. LangChain Chroma 통합 인스턴스 생성
            self.collection_name = collection_name
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            print(f"Chroma 벡터 저장소 설정 완료. 컬렉션: '{self.collection_name}'")
            
            # 4. Retriever 설정
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        except Exception as e:
            print(f"RAGModule 초기화 중 심각한 오류 발생: {e}")
            # 초기화 실패 시, 모듈의 핵심 기능이 작동하지 않도록 설정
            self.embedding_function = None
            self.chroma_client = None
            self.vector_store = None
            self.retriever = None
            raise

    def search(self, query: str, k: int = 3) -> list:
        """유사도 검색을 수행하고 결과를 포맷팅하여 반환합니다."""
        if not self.retriever:
            print("RAG 모듈이 제대로 초기화되지 않아 검색을 수행할 수 없습니다.")
            return []
            
        try:
            print(f"RAG 검색 수행: '{query}'")
            # LangChain 0.2.x 부터 get_relevant_documents가 기본 검색 메서드
            results = self.retriever.get_relevant_documents(query)
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })
            print(f"{len(formatted_results)}개의 유사 상품을 찾았습니다.")
            return formatted_results
        except Exception as e:
            print(f"RAG 검색 중 오류 발생: {e}")
            return []

def get_rag_module(collection_name="products"):
    """
    RAGModule의 싱글턴 인스턴스를 가져옵니다.
    필요한 경우에만 인스턴스를 생성합니다.
    """
    # 이 예제에서는 단순 생성을 사용하지만, 필요시 실제 싱글턴 패턴 적용 가능
    print(f"[RAG Module] RAGModule 인스턴스('{collection_name}' 컬렉션) 생성을 요청받았습니다.")
    return RAGModule(collection_name=collection_name)


# --- 모듈 임포트 시 자동 실행되던 인스턴스 생성 코드 제거 ---
# print("[RAG Module] 기본 RAGModule 인스턴스('products' 컬렉션) 생성을 시도합니다...")
# RAG_MODULE = RAGModule(collection_name="products")
# print("[RAG Module] 기본 RAGModule 인스턴스 생성 완료.")

# def get_rag_module():
#     return RAG_MODULE 