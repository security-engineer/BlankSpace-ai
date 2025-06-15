from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
import os

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

    def handle_conversation_turn(self, session_id: str, user_message: str, page_info: dict) -> dict:
        """
        한 턴의 대화를 처리하고, 추천 상품 정보(메타데이터)를 함께 반환합니다.
        """
        try:
            # RAG로 관련 상품 검색
            search_query = f"Page Context: {page_info.get('pageTitle', 'N/A')}. User Query: {user_message}"
            retrieved_docs = self.rag_module.search(search_query)
            
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