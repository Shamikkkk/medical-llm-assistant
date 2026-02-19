from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.history import get_session_history


def build_rag_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a clinical assistant specializing in cardiovascular medicine. "
                "You also handle cardiopulmonary overlap topics (e.g., COPD, pulmonary "
                "hypertension) only from a cardiovascular relevance perspective. "
                "Use ONLY the provided abstracts to answer. "
                "If the abstracts do not contain enough evidence, reply: "
                "'Insufficient evidence in the provided abstracts.' "
                "Do not invent PMIDs and cite only PMIDs present in the abstracts. "
                "Do NOT include a 'Reframe:' section. "
                "Do NOT describe how you interpreted or reframed the question. "
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
        | retriever
        | RunnableLambda(_format_docs)
    )
    return RunnablePassthrough.assign(context=context_chain) | prompt | llm


def build_chat_chain(base_chain):
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


def _format_docs(docs: list) -> str:
    sections = []
    for doc in docs:
        meta = doc.metadata or {}
        pmid = meta.get("pmid", "")
        title = meta.get("title", "")
        journal = meta.get("journal", "")
        year = meta.get("year", "")
        abstract = doc.page_content or ""
        if title and abstract.lower().startswith(str(title).lower()):
            abstract = abstract[len(str(title)) :].lstrip()
        sections.append(
            "PMID: {pmid}\nTitle: {title}\nJournal: {journal}\nYear: {year}\n"
            "Abstract: {abstract}".format(
                pmid=pmid,
                title=title,
                journal=journal,
                year=year,
                abstract=abstract,
            )
        )
    return "\n\n---\n\n".join(sections)
