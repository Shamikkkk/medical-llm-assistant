from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.history import get_session_history
from src.core.retrieval import build_context_text

try:  # pragma: no cover - import path varies slightly across langchain-core versions
    from langchain_core.runnables.base import Runnable as _RunnableType
except Exception:  # pragma: no cover
    _RunnableType = None


def _coerce_retriever_to_runnable(retriever):
    if _RunnableType is not None and isinstance(retriever, _RunnableType):
        return retriever
    if hasattr(retriever, "get_relevant_documents"):
        return RunnableLambda(lambda query: retriever.get_relevant_documents(query))
    if callable(retriever):
        return RunnableLambda(lambda query: retriever(query))
    if hasattr(retriever, "retrieve"):
        return RunnableLambda(lambda query: retriever.retrieve(query))
    if hasattr(retriever, "invoke"):
        return RunnableLambda(lambda query: retriever.invoke(query))
    raise TypeError(f"Retriever cannot be coerced to Runnable: {type(retriever)}")


def build_rag_chain(
    llm,
    retriever,
    *,
    max_abstracts: int = 8,
    max_context_tokens: int = 2500,
    trim_strategy: str = "truncate",
):
    retriever_runnable = _coerce_retriever_to_runnable(retriever)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a clinical literature assistant for biomedical and public-health topics. "
                "Use ONLY the provided abstracts to answer. "
                "If the abstracts do not contain enough evidence, reply: "
                "'Insufficient evidence in the provided abstracts.' "
                "Do not claim access to full-text content unless it is explicitly present in the provided context. "
                "Do not invent PMIDs and cite only PMIDs present in the abstracts. "
                "Do NOT include a 'Reframe:' section. "
                "Do NOT describe how you interpreted or reframed the question. "
                "Answer the user's current question explicitly and avoid vague phrasing. "
                "If the question is a follow-up, briefly anchor to prior context in one sentence. "
                "Answer directly and cite PMIDs. "
                "Use this structure: "
                "(1) 1-2 short paragraphs with a direct answer. "
                "(2) Evidence summary as 2-5 bullet points with PMID citations. "
                "(3) Optional caveats/limitations if evidence is weak.",
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                "Question: {input}\n\nAbstracts:\n{context}",
            ),
        ]
    )

    context_chain = (
        RunnableLambda(lambda x: x["retrieval_query"])
        | retriever_runnable
        | RunnableLambda(
            lambda docs: _format_docs(
                docs,
                max_abstracts=max_abstracts,
                max_context_tokens=max_context_tokens,
                trim_strategy=trim_strategy,
            )
        )
    )
    return RunnablePassthrough.assign(context=context_chain) | prompt | llm


def build_chat_chain(base_chain):
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Unique identifier for the chat session.",
                default="",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="branch_id",
                annotation=str,
                name="Branch ID",
                description="Unique identifier for the active conversation branch.",
                default="main",
                is_shared=True,
            ),
        ],
    )


def _format_docs(
    docs: list,
    *,
    max_abstracts: int = 8,
    max_context_tokens: int = 2500,
    trim_strategy: str = "truncate",
) -> str:
    return build_context_text(
        docs,
        max_abstracts=max_abstracts,
        max_context_tokens=max_context_tokens,
        trim_strategy=trim_strategy,
    )
