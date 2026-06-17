"""Chat API with SSE streaming."""
import json
import uuid
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.chat import ChatSession, ChatMessage
from app.models.document import DocumentChunk
from app.schemas.chat import (
    ChatSessionOut,
    ChatSessionDetailOut,
    ChatMessageOut,
    CreateSessionRequest,
    SendMessageRequest,
)
from app.services.agents.mentor_agent import MentorState
from app.services.rag.retriever import hybrid_search

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/sessions", response_model=ChatSessionOut, status_code=201)
async def create_session(body: CreateSessionRequest, db: AsyncSession = Depends(get_db)):
    session = ChatSession(title=body.title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ChatSession).order_by(ChatSession.updated_at.desc()))
    return result.scalars().all()


@router.get("/sessions/{session_id}", response_model=ChatSessionDetailOut)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id)
        .options(selectinload(ChatSession.messages))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await db.get(ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    await db.delete(session)
    await db.commit()


@router.post("/sessions/{session_id}/messages")
async def send_message(
    session_id: uuid.UUID,
    body: SendMessageRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.id == session_id)
        .options(selectinload(ChatSession.messages))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")

    # Auto-title from first message
    if len(session.messages) == 0 and session.title == "New Chat":
        session.title = body.content[:60] + ("..." if len(body.content) > 60 else "")

    # Build LangChain history
    history = []
    for msg in session.messages[-10:]:  # last 10 messages for context window
        if msg.role == "user":
            history.append(HumanMessage(content=msg.content))
        else:
            history.append(AIMessage(content=msg.content))

    # Save user message
    user_msg = ChatMessage(session_id=session_id, role="user", content=body.content)
    db.add(user_msg)
    await db.commit()

    mentor_graph = request.app.state.mentor_graph

    async def stream_response() -> AsyncGenerator[str, None]:
        state = MentorState(
            session_id=str(session_id),
            user_message=body.content,
            history=history,
            search_query="",
            retrieved_chunks=[],
            response_text="",
            cited_chunk_ids=[],
        )

        final_state: dict = {}
        response_text = ""
        try:
            # Real token streaming: stream_mode="messages" yields LLM tokens from
            # inside nodes as they are generated; "values" yields full state
            # snapshots (the last one is the final state with citations).
            async for mode, chunk in mentor_graph.astream(
                state, stream_mode=["messages", "values"]
            ):
                if mode == "messages":
                    msg, meta = chunk
                    # only the answer node streams to the user; the query-rewrite
                    # node ("contextualize") also calls the LLM — skip its tokens
                    if meta.get("langgraph_node") == "generate" and msg.content:
                        response_text += msg.content
                        yield f"data: {json.dumps({'type': 'token', 'text': msg.content})}\n\n"
                elif mode == "values":
                    final_state = chunk
        except Exception as e:
            logger.exception("Mentor agent error: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'text': 'Failed to generate response.'})}\n\n"
            return

        response_text = final_state.get("response_text") or response_text
        cited_ids = final_state.get("cited_chunk_ids", [])
        chunks = final_state.get("retrieved_chunks", [])

        # Build citations for response
        citations = []
        seen_ids = set()
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if cid and cid in cited_ids and cid not in seen_ids:
                seen_ids.add(cid)
                heading = " > ".join(
                    filter(None, [chunk.get("h1_title"), chunk.get("h2_title"), chunk.get("h3_title")])
                )
                citations.append({
                    "chunk_id": cid,
                    "filename": chunk.get("filename", ""),
                    "heading": heading or None,
                })

        yield f"data: {json.dumps({'type': 'done', 'citations': citations})}\n\n"

        # Persist assistant message
        async with db.begin_nested():
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=response_text,
                source_chunks=[uuid.UUID(cid) for cid in cited_ids if cid],
            )
            db.add(assistant_msg)
        await db.commit()

    return StreamingResponse(stream_response(), media_type="text/event-stream")
