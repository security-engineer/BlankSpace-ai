from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import os
import re

from rag_module import get_rag_module

class ConversationManager:
    def __init__(self):
        """
        대화 관리자를 초기화합니다.
        LLM 모델, 프롬프트 템플릿, RAG 모듈을 설정합니다.
        """
        try:
            # get_rag_module 함수를 호출하여 RAG 모듈 인스턴스를 가져옵니다.
            self.rag_module = get_rag_module(collection_name="products")
            print("[ConversationManager] RAG 모듈 인스턴스 획득 성공.")
        except Exception as e:
            print(f"RAG 모듈 초기화 중 심각한 오류 발생: {e}")
            # RAG 모듈 없이는 대화 관리자를 사용할 수 없으므로 에러를 발생시킵니다.
            raise RuntimeError("RAG 모듈이 초기화되지 않았습니다. ConversationManager를 사용할 수 없습니다.") from e
        
        # LLM 모델 설정
        llm_host = os.getenv("OLLAMA_HOST", "ollama_server")
        llm_port = os.getenv("OLLAMA_PORT", "11434")
        llm_model = os.getenv("OLLAMA_MODEL", "gemma3:12b-it-qat")
        
        # 사용자 요청에 따라 gemma3:12b-it-qat 모델을 사용
        self.llm = ChatOllama(
            model=llm_model,
            base_url=f"http://{llm_host}:{llm_port}",
            temperature=0.7,
        )
        print(f"ChatOllama 모델('{llm_model}') 설정 완료.")

        # 대화 기록을 세션별로 저장
        self.session_histories = {}

        # 프롬프트 템플릿 설정
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a helpful and friendly AI shopping assistant for 'BlankSpace'. "
             "Your goal is to help users find the best products based on their needs. "
             "Use the provided context from our product database to give specific recommendations. "
             "Always respond in Korean."
             "\n--- CONTEXT FROM PRODUCT DATABASE ---\n"
             "{context}"
             "\n--- END OF CONTEXT ---"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        self.chain = self.prompt_template | self.llm
        print("LangChain 체인 설정 완료.")

        # 키워드-카테고리 매핑 테이블 초기화
        self.category_mapping = {
            'necklace': '목걸이',
            'chain': '체인',
            'belt': '벨트',
            'shirt': '셔츠',
            'pants': '바지',
            'accessory': '악세사리',
            'acc': '악세사리',
            'shoes': '신발',
            'hat': '모자',
            'bag': '가방',
            'outer': '아우터',
            'jacket': '자켓',
            'coat': '코트',
        }

    def _get_chat_history(self, session_id: str):
        """세션 ID에 해당하는 대화 기록을 가져옵니다."""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        return self.session_histories[session_id]

    def clear_session(self, session_id: str):
        """특정 세션의 대화 기록을 초기화합니다."""
        if session_id in self.session_histories:
            self.session_histories[session_id] = []
            print(f"세션 {session_id}의 대화 기록이 초기화되었습니다.")

    def extract_keywords(self, text):
        """사용자 메시지에서 검색 키워드를 추출합니다."""
        # 한국어 키워드 사전 (쇼핑 관련 키워드)
        korean_keywords = {
            '목걸이', '귀걸이', '반지', '팔찌', '시계', '모자', '벨트', '신발', 
            '바지', '셔츠', '자켓', '코트', '가방', '지갑', '안경', '선글라스',
            '악세사리', '액세서리', '패션', '의류'
        }
        
        # 사용자 메시지에서 키워드 추출
        keywords = []
        
        # 1. 한국어 키워드 매칭
        for keyword in korean_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        # 2. 영어 키워드 매칭 (영어-한국어 매핑 테이블 활용)
        for eng_key, kor_value in self.category_mapping.items():
            if eng_key.lower() in text.lower():
                keywords.append(eng_key)
                if kor_value not in keywords:
                    keywords.append(kor_value)
        
        print(f"추출된 키워드: {keywords}")
        return keywords

    def keyword_search(self, keyword):
        """키워드 기반으로 상품을 검색합니다."""
        results = []
        
        try:
            # 임베딩 검색을 활용하여 키워드로 검색
            keyword_results = self.rag_module.search(keyword)
            if keyword_results:
                results.extend(keyword_results)
                print(f"키워드 '{keyword}'로 {len(keyword_results)}개 상품 찾음")
        except Exception as e:
            print(f"키워드 검색 중 오류: {e}")
        
        return results

    def handle_conversation_turn(self, session_id: str, user_message: str, page_info: dict) -> dict:
        """
        한 턴의 대화를 처리하고, 추천 상품 정보(메타데이터)를 함께 반환합니다.
        """
        try:
            # 1. 검색 쿼리 최적화: 사용자 메시지에 더 가중치를 둠
            # 기존: search_query = f"Page Context: {page_info.get('pageTitle', 'N/A')}. User Query: {user_message}"
            search_query = user_message  # 사용자 메시지만으로 검색
            
            # 2. RAG로 관련 상품 검색
            retrieved_docs = self.rag_module.search(search_query)
            print(f"RAG 검색 결과: {len(retrieved_docs)}개 문서 발견")
            
            # 3. 검색 결과가 부족한 경우 키워드 기반 보조 검색 수행
            if len(retrieved_docs) < 2:
                print("검색 결과가 부족하여 키워드 기반 검색 시도...")
                keywords = self.extract_keywords(user_message)
                for keyword in keywords:
                    keyword_results = self.keyword_search(keyword)
                    if keyword_results:
                        # 중복 제거하면서 결과 추가
                        existing_ids = {doc.get('metadata', {}).get('product_id') for doc in retrieved_docs}
                        for doc in keyword_results:
                            doc_id = doc.get('metadata', {}).get('product_id')
                            if doc_id and doc_id not in existing_ids:
                                retrieved_docs.append(doc)
                                existing_ids.add(doc_id)
                        print(f"키워드 '{keyword}'로 추가 검색 후 총 {len(retrieved_docs)}개 문서")
                        break
                        
            # 추천된 상품의 메타데이터를 저장할 리스트
            recommended_products = [doc.get('metadata', {}) for doc in retrieved_docs]

            # LLM에게 전달할 컨텍스트 생성
            context_parts = []
            for doc in retrieved_docs:
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                meta_str = ", ".join([f"{key}: {value}" for key, value in metadata.items()])
                context_parts.append(f"상품 정보: {content} (상세: {meta_str})")
            
            context_str = "\n\n".join(context_parts)
            if not context_str:
                context_str = "검색된 상품 정보가 없습니다. 일반적인 대화를 나눠주세요."

            # LLM 호출
            chat_history = self._get_chat_history(session_id)
            
            response = self.chain.invoke({
                "context": context_str,
                "chat_history": chat_history,
                "input": user_message,
            })
            
            ai_response_text = response.content

            # 대화 기록 업데이트
            chat_history.extend([
                HumanMessage(content=user_message),
                AIMessage(content=ai_response_text),
            ])
            self.session_histories[session_id] = chat_history

            # 최종 응답 구성: 메시지와 추천 상품 메타데이터 포함
            final_action = {
                "action": "recommend_products",
                "payload": {"products": recommended_products}
            } if recommended_products else None

            return {
                "message": ai_response_text,
                "action": final_action
            }

        except Exception as e:
            print(f"대화 처리 중 오류 발생: {e}")
            return {"message": "죄송합니다, AI 어시스턴트 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", "action": None}

# 싱글턴 인스턴스 생성
try:
    conversation_manager_instance = ConversationManager()
except Exception as e:
    print(f"ConversationManager 인스턴스 생성 실패. AI 기능이 비활성화될 수 있습니다. 에러: {e}")
    conversation_manager_instance = None 