"""Response generation using Azure OpenAI / OpenAI with supply chain context.

Configures the LLM with a domain-specific system prompt and formats
retrieved context for grounded answer generation.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

from config.settings import get_settings

SYSTEM_PROMPT = """You are an expert supply chain and logistics analyst for a global logistics company. Your role is to provide accurate, data-driven answers based on the provided context documents.

Guidelines:
- Base your answers strictly on the provided context. If the context doesn't contain enough information, say so clearly.
- When citing data, reference the source document and page number when available.
- For numerical data, provide specific figures from the documents rather than generalizations.
- Structure complex answers with clear headings and bullet points.
- Flag any potential compliance or regulatory implications when relevant.
- If data from multiple sources conflicts, highlight the discrepancy.

Always end your response with a "Sources" section listing the documents used."""


class ResponseGenerator:
    """Generate responses grounded in retrieved supply chain documents.

    Automatically selects Azure OpenAI or standard OpenAI based on
    available configuration.
    """

    def __init__(self, model: Optional[BaseChatModel] = None):
        self.settings = get_settings()
        self.llm = model or self._create_llm()

    def _create_llm(self) -> BaseChatModel:
        """Initialize the language model based on configuration."""
        if self.settings.use_azure_openai:
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI(
                azure_deployment=self.settings.azure_openai_deployment_name,
                azure_endpoint=self.settings.azure_openai_endpoint,
                api_key=self.settings.azure_openai_api_key,
                api_version=self.settings.azure_openai_api_version,
                temperature=0.1,
            )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o",
            api_key=self.settings.openai_api_key,
            temperature=0.1,
        )

    def generate(
        self,
        query: str,
        context_documents: list[Document],
    ) -> str:
        """Generate a response grounded in the provided context."""
        context_str = self._format_context(context_documents)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {query}"),
        ]

        response = self.llm.invoke(messages)
        return str(response.content)

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not documents:
            return "No relevant documents found."

        parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            score = doc.metadata.get("retrieval_score", "")

            header = f"[Document {i}] Source: {source}"
            if page:
                header += f", Page {page}"
            if score:
                header += f" (relevance: {score:.3f})"

            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)
