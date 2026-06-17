"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { MessageSquare, Plus, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { ChatSession } from "@/lib/types";
import { formatDate } from "@/lib/utils";

export default function ChatPage() {
  const router = useRouter();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.chat.listSessions().then(setSessions).finally(() => setLoading(false));
  }, []);

  async function createSession() {
    const session = await api.chat.createSession();
    router.push(`/chat/${session.id}`);
  }

  async function deleteSession(id: string, e: React.MouseEvent) {
    e.preventDefault();
    await api.chat.deleteSession(id);
    setSessions((prev) => prev.filter((s) => s.id !== id));
  }

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-white">AI Mentor Chat</h1>
        <button
          onClick={createSession}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {loading ? (
        <div className="text-gray-500 text-sm">Loading sessions...</div>
      ) : sessions.length === 0 ? (
        <div className="flex flex-col items-center justify-center flex-1 text-center gap-4">
          <MessageSquare className="w-12 h-12 text-gray-700" />
          <div>
            <p className="text-gray-400 font-medium">No chats yet</p>
            <p className="text-gray-600 text-sm mt-1">
              Start a new conversation with your AI mentor
            </p>
          </div>
          <button
            onClick={createSession}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded-lg transition-colors"
          >
            Start First Chat
          </button>
        </div>
      ) : (
        <div className="grid gap-3">
          {sessions.map((session) => (
            <a
              key={session.id}
              href={`/chat/${session.id}`}
              className="flex items-center justify-between p-4 bg-gray-900 border border-gray-800 rounded-xl hover:border-gray-700 transition-colors group"
            >
              <div className="flex items-center gap-3 min-w-0">
                <MessageSquare className="w-4 h-4 text-emerald-400 shrink-0" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-gray-200 truncate">{session.title}</p>
                  <p className="text-xs text-gray-600">{formatDate(session.updated_at)}</p>
                </div>
              </div>
              <button
                onClick={(e) => deleteSession(session.id, e)}
                className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-600 hover:text-red-400 transition-all rounded"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
