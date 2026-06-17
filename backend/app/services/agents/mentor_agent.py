"""Mentor LangGraph agent — answers questions using RAG with citations."""
from typing import TypedDict, Annotated
import operator
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.services.rag.retriever import hybrid_search
from app.services.agents.llm_factory import get_llm

SYSTEM_PROMPT = """You are an expert AI learning mentor. Your role is to answer questions based on the user's personal knowledge base.

When answering:
1. Use ONLY information from the provided context chunks
2. Cite sources using [1], [2] notation referring to the numbered chunks
3. If the context doesn't contain enough information, say so honestly
4. Adapt your explanation complexity to the user's apparent level
5. Be concise but thorough

Context:
{context}"""


CONTEXTUALIZE_PROMPT = """Given the conversation so far and the user's latest message,
rewrite the latest message into a STANDALONE search query understandable without
the conversation. Resolve pronouns and references ("it", "that", "this") to the
concrete topic discussed. Output ONLY the rewritten query — no preamble, no answer.
If the latest message is already self-contained, return it unchanged.

Conversation:
{conversation}

Latest message: {message}

Standalone search query:"""


class MentorState(TypedDict):
    session_id: str
    user_message: str
    history: list[BaseMessage]
    search_query: str
    retrieved_chunks: list[dict]
    response_text: str
    cited_chunk_ids: list[str]


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        heading = " > ".join(
            filter(None, [chunk.get("h1_title"), chunk.get("h2_title"), chunk.get("h3_title")])
        )
        source = chunk.get("filename", "unknown")
        if chunk.get("page_number"):
            source += f" (page {chunk['page_number']})"
        header = f"[{i}] {source}" + (f" — {heading}" if heading else "")
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


async def contextualize_query(state: MentorState) -> dict:
    """Rewrite a follow-up into a standalone search query using chat history.

    Without this, retrieval sees raw "tell me more about it" — pronouns carry no
    topic, so the search returns garbage. We only call the LLM when there IS
    history (the first message is already self-contained → skip the latency).
    """
    history = state["history"]
    message = state["user_message"]
    if not history:
        return {"search_query": message}

    conversation = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history[-6:]
    )
    llm = get_llm(streaming=False)
    prompt = CONTEXTUALIZE_PROMPT.format(conversation=conversation, message=message)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    query = (response.content or "").strip() or message
    return {"search_query": query}


async def retrieve_context(state: MentorState) -> dict:
    chunks = await hybrid_search(state["search_query"])
    return {"retrieved_chunks": chunks}


async def generate_response(state: MentorState) -> dict:
    chunks = state["retrieved_chunks"]
    context = _format_context(chunks) if chunks else "No relevant content found in your knowledge base."

    system = SystemMessage(content=SYSTEM_PROMPT.format(context=context))
    messages = [system] + state["history"] + [HumanMessage(content=state["user_message"])]

    # Stream tokens: with streaming=True + astream, each chunk is surfaced by
    # LangGraph's stream_mode="messages" so the API can forward real tokens.
    llm = get_llm(streaming=True)
    text = ""
    async for chunk in llm.astream(messages):
        text += chunk.content or ""

    # Extract cited chunk IDs for chunks referenced in the response
    cited: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        if f"[{i}]" in text and chunk.get("chunk_id"):
            cited.append(chunk["chunk_id"])

    return {"response_text": text, "cited_chunk_ids": cited}


def build_mentor_graph():
    graph = StateGraph(MentorState)
    graph.add_node("contextualize", contextualize_query)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", generate_response)
    graph.set_entry_point("contextualize")
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
