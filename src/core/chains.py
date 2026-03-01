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


SYSTEM_PROMPT = """You are a rigorous clinical literature assistant. Your sole source of truth is the numbered list of PubMed abstracts provided in the context. Follow these rules without exception:

STRUCTURE (always use this exact structure):
## Direct Answer
One to three sentences directly answering the question. Be concrete. Do not hedge unless the evidence genuinely conflicts.

## Evidence Summary
2-5 bullet points. Each bullet MUST end with the PMID in brackets, e.g. [PMID: 12345678].
Only cite PMIDs present in the provided abstracts. Never invent or guess a PMID.
Each bullet should state a specific finding, not a vague generality.

## Evidence Quality
One sentence assessing overall evidence quality: label it as one of - Strong (>=3 concordant RCTs), Moderate (observational or mixed), Preliminary (case reports or single studies), or Insufficient (no relevant abstracts).

## Caveats (optional)
Include only if evidence is weak, contradictory, or the question is outside scope.

STRICT RULES:
- Never say "I don't have access to full text" - just use what is in the abstracts.
- Never include a "Reframe:" section.
- Never describe your reasoning process to the user.
- If the abstracts contain no relevant information, respond only with: "The provided abstracts do not contain sufficient evidence to answer this question."
- Keep the total response under 450 words unless evidence complexity genuinely requires more.
- Use plain, clinician-accessible language. Avoid jargon unless it is in the source material.
"""


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
                SYSTEM_PROMPT,
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
