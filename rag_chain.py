"""
rag_chain.py
LangChain-based enterprise RAG pipeline with observability hooks.
"""

import time
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


ENTERPRISE_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an enterprise AI assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough information."
Never make up facts. Never reveal PII.

Context:
{context}

Question: {question}

Answer:"""
)


class EnterpriseRAGChain:
    """
    Production RAG chain with:
    - LangChain RetrievalQA
    - OpenAI GPT-4 as the LLM
    - FAISS as the vector store
    - Built-in latency and token tracking
    """

    def __init__(
        self,
        index_path: str,
        model: str = "gpt-4",
        top_k: int = 5,
        temperature: float = 0.0,
    ):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(index_path, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": ENTERPRISE_PROMPT}
        )
        self._query_log = []

    def query(self, question: str, user_id: Optional[str] = None) -> dict:
        """Run a RAG query and return answer + sources + metrics."""
        start = time.time()
        result = self.chain.invoke({"query": question})
        latency_ms = round((time.time() - start) * 1000, 2)

        sources = [
            {
                "content": doc.page_content[:300],
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", None),
            }
            for doc in result.get("source_documents", [])
        ]

        log_entry = {
            "user_id": user_id,
            "question": question,
            "latency_ms": latency_ms,
            "num_sources": len(sources),
        }
        self._query_log.append(log_entry)

        return {
            "answer": result["result"],
            "sources": sources,
            "latency_ms": latency_ms,
            "model": self.llm.model_name,
        }

    def get_query_logs(self) -> list:
        return self._query_log
