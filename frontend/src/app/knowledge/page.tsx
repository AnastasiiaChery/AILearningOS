"use client";

import { useEffect, useRef, useState } from "react";
import { BookOpen, CheckCircle, Clock, Upload, XCircle, Trash2, FileText } from "lucide-react";
import { api } from "@/lib/api";
import type { Document } from "@/lib/types";
import { cn, formatDate, formatFileSize } from "@/lib/utils";

const statusIcon = {
  pending: <Clock className="w-4 h-4 text-yellow-500" />,
  processing: <Clock className="w-4 h-4 text-blue-400 animate-spin" />,
  ready: <CheckCircle className="w-4 h-4 text-emerald-500" />,
  error: <XCircle className="w-4 h-4 text-red-500" />,
};

const statusLabel = {
  pending: "Pending",
  processing: "Processing...",
  ready: "Ready",
  error: "Error",
};

export default function KnowledgePage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    loadDocuments();
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  function startPolling() {
    if (pollingRef.current) return;
    pollingRef.current = setInterval(async () => {
      const docs = await api.knowledge.list();
      setDocuments(docs);
      const hasPending = docs.some((d) => d.status === "pending" || d.status === "processing");
      if (!hasPending && pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }, 2000);
  }

  async function loadDocuments() {
    const docs = await api.knowledge.list();
    setDocuments(docs);
    if (docs.some((d) => d.status === "pending" || d.status === "processing")) {
      startPolling();
    }
  }

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    try {
      for (const file of Array.from(files)) {
        await api.knowledge.upload(file);
      }
      await loadDocuments();
      startPolling();
    } catch (e: unknown) {
      alert((e as Error).message || "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function deleteDoc(id: string) {
    if (!confirm("Delete this document and all its chunks?")) return;
    await api.knowledge.delete(id);
    setDocuments((prev) => prev.filter((d) => d.id !== id));
  }

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-white">Knowledge Base</h1>
        <span className="text-sm text-gray-500">{documents.length} documents</span>
      </div>

      {/* Upload Zone */}
      <div
        className={cn(
          "border-2 border-dashed rounded-xl p-8 text-center mb-6 transition-colors cursor-pointer",
          dragOver ? "border-emerald-500 bg-emerald-500/5" : "border-gray-800 hover:border-gray-700"
        )}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleUpload(e.dataTransfer.files);
        }}
      >
        <Upload className="w-8 h-8 text-gray-600 mx-auto mb-3" />
        <p className="text-sm text-gray-400 font-medium">
          {uploading ? "Uploading..." : "Drop files or click to upload"}
        </p>
        <p className="text-xs text-gray-600 mt-1">Supports Markdown (.md) and PDF files</p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".md,.markdown,.pdf"
          multiple
          className="hidden"
          onChange={(e) => handleUpload(e.target.files)}
        />
      </div>

      {/* Document List */}
      {documents.length === 0 ? (
        <div className="flex flex-col items-center justify-center flex-1 text-center gap-3">
          <BookOpen className="w-12 h-12 text-gray-700" />
          <p className="text-gray-400 font-medium">No documents yet</p>
          <p className="text-gray-600 text-sm">Upload markdown notes or PDF documents to get started</p>
        </div>
      ) : (
        <div className="grid gap-3">
          {documents.map((doc) => (
            <div
              key={doc.id}
              className="flex items-center gap-4 p-4 bg-gray-900 border border-gray-800 rounded-xl group"
            >
              <FileText className="w-5 h-5 text-gray-600 shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-200 truncate">{doc.original_filename}</p>
                <div className="flex items-center gap-3 mt-0.5">
                  <span className="text-xs text-gray-600">{formatFileSize(doc.file_size)}</span>
                  {doc.chunk_count != null && (
                    <span className="text-xs text-gray-600">{doc.chunk_count} chunks</span>
                  )}
                  <span className="text-xs text-gray-600">{formatDate(doc.created_at)}</span>
                </div>
                {doc.error_msg && (
                  <p className="text-xs text-red-400 mt-0.5 truncate">{doc.error_msg}</p>
                )}
              </div>
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  {statusIcon[doc.status]}
                  <span className="text-xs text-gray-500">{statusLabel[doc.status]}</span>
                </div>
                <button
                  onClick={() => deleteDoc(doc.id)}
                  className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-600 hover:text-red-400 transition-all rounded"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
