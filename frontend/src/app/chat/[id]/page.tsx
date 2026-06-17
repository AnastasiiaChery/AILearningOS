"use client";

import { useEffect, useRef, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowLeft, Send, ExternalLink, FileText } from "lucide-react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "@/lib/api";
import { readSSEStream } from "@/lib/streaming";
import type { ChatMessage, ChatSessionDetail, Citation } from "@/lib/types";
import { cn } from "@/lib/utils";

interface MessageWithCitations extends ChatMessage {
  citations?: Citation[];
  streaming?: boolean;
}

export default function ChatSessionPage() {
  const { id } = useParams<{ id: string }>();
  const [session, setSession] = useState<ChatSessionDetail | null>(null);
  const [messages, setMessages] = useState<MessageWithCitations[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.chat.getSession(id).then((s) => {
      setSession(s);
      setMessages(s.messages);
    });
  }, [id]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    if (!input.trim() || sending) return;
    const content = input.trim();
    setInput("");
    setSending(true);

    // Optimistic user message
    const userMsg: MessageWithCitations = {
      id: crypto.randomUUID(),
      session_id: id,
      role: "user",
      content,
      source_chunks: null,
      created_at: new Date().toISOString(),
    };
    const streamId = crypto.randomUUID();
    const assistantMsg: MessageWithCitations = {
      id: streamId,
      session_id: id,
      role: "assistant",
      content: "",
      source_chunks: null,
      created_at: new Date().toISOString(),
      streaming: true,
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    try {
      const response = await api.chat.streamMessage(id, content);
      let fullText = "";
      let citations: Citation[] = [];

      for await (const event of readSSEStream(response)) {
        if (event.type === "token") {
          fullText += event.text;
          setMessages((prev) =>
            prev.map((m) => (m.id === streamId ? { ...m, content: fullText } : m))
          );
        } else if (event.type === "done") {
          citations = event.citations;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === streamId ? { ...m, content: fullText, citations, streaming: false } : m
            )
          );
        } else if (event.type === "error") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === streamId
                ? { ...m, content: "Sorry, an error occurred. Please try again.", streaming: false }
                : m
            )
          );
        }
      }
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === streamId
            ? { ...m, content: "Connection error. Please try again.", streaming: false }
            : m
        )
      );
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-800 bg-gray-950">
        <Link href="/chat" className="p-1.5 text-gray-500 hover:text-gray-200 rounded transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <span className="text-sm font-medium text-gray-200 truncate">
          {session?.title || "Loading..."}
        </span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center gap-3 text-gray-600">
            <p className="text-sm">Ask your AI mentor anything from your knowledge base.</p>
            <p className="text-xs">
              Try: <em>"What is dependency injection?"</em> or{" "}
              <em>"Explain async/await simply"</em>
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={cn("flex", msg.role === "user" ? "justify-end" : "justify-start")}>
            <div className={cn("max-w-[80%]", msg.role === "user" ? "ml-12" : "mr-12")}>
              <div
                className={cn(
                  "rounded-2xl px-4 py-3 text-sm",
                  msg.role === "user"
                    ? "bg-emerald-600 text-white rounded-br-sm"
                    : "bg-gray-800 text-gray-100 rounded-bl-sm"
                )}
              >
                {msg.role === "assistant" ? (
                  <div className="prose-chat">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    {msg.streaming && (
                      <span className="inline-block w-2 h-4 bg-emerald-400 ml-1 animate-pulse rounded-sm" />
                    )}
                  </div>
                ) : (
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                )}
              </div>

              {/* Citations */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 space-y-1">
                  <p className="text-xs text-gray-600 px-1">Sources:</p>
                  {msg.citations.map((c, i) => (
                    <div key={c.chunk_id} className="flex items-center gap-2 px-2 py-1 bg-gray-900 border border-gray-800 rounded-lg text-xs text-gray-400">
                      <FileText className="w-3 h-3 text-emerald-500 shrink-0" />
                      <span className="truncate">
                        [{i + 1}] {c.filename}
                        {c.heading && <span className="text-gray-600"> — {c.heading}</span>}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-gray-800 bg-gray-950">
        <div className="flex items-end gap-2 bg-gray-900 border border-gray-800 rounded-xl p-2 focus-within:border-gray-700 transition-colors">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask your AI mentor..."
            rows={1}
            className="flex-1 bg-transparent text-sm text-gray-100 placeholder-gray-600 resize-none outline-none px-2 py-1 max-h-32"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || sending}
            className="p-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded-lg transition-colors shrink-0"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        <p className="text-xs text-gray-700 mt-1.5 px-1">Shift+Enter for new line</p>
      </div>
    </div>
  );
}
