"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { HelpCircle, Plus, X } from "lucide-react";
import { api } from "@/lib/api";
import type { Document, Quiz } from "@/lib/types";
import { formatDate, cn } from "@/lib/utils";

export default function QuizzesPage() {
  const router = useRouter();
  const [quizzes, setQuizzes] = useState<Quiz[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [questionCount, setQuestionCount] = useState(5);
  const [difficulty, setDifficulty] = useState("medium");
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    api.quizzes.list().then(setQuizzes);
    api.knowledge.list().then((docs) => setDocuments(docs.filter((d) => d.status === "ready")));
  }, []);

  async function generateQuiz() {
    if (!selectedDoc) return;
    setGenerating(true);
    try {
      const quiz = await api.quizzes.generate({ document_id: selectedDoc, question_count: questionCount, difficulty });
      router.push(`/quizzes/${quiz.id}`);
    } catch (e: unknown) {
      alert((e as Error).message || "Failed to generate quiz");
      setGenerating(false);
    }
  }

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-white">Quizzes</h1>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          Generate Quiz
        </button>
      </div>

      {quizzes.length === 0 ? (
        <div className="flex flex-col items-center justify-center flex-1 text-center gap-4">
          <HelpCircle className="w-12 h-12 text-gray-700" />
          <p className="text-gray-400 font-medium">No quizzes yet</p>
          <p className="text-gray-600 text-sm">Generate a quiz from any document in your knowledge base</p>
        </div>
      ) : (
        <div className="grid gap-3">
          {quizzes.map((quiz) => (
            <a
              key={quiz.id}
              href={`/quizzes/${quiz.id}`}
              className="block p-4 bg-gray-900 border border-gray-800 rounded-xl hover:border-gray-700 transition-colors"
            >
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-gray-200">{quiz.title}</p>
                <span className="text-xs text-gray-600">{formatDate(quiz.created_at)}</span>
              </div>
            </a>
          ))}
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-white">Generate Quiz</h2>
              <button onClick={() => setShowModal(false)} className="text-gray-500 hover:text-gray-200">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-4 mb-4">
              <div>
                <label className="text-xs text-gray-500 mb-1.5 block">Source Document</label>
                <select
                  value={selectedDoc}
                  onChange={(e) => setSelectedDoc(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 outline-none focus:border-emerald-500"
                >
                  <option value="">Select a document...</option>
                  {documents.map((doc) => (
                    <option key={doc.id} value={doc.id}>{doc.original_filename}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-500 mb-1.5 block">Difficulty</label>
                <div className="grid grid-cols-3 gap-2">
                  {["easy", "medium", "hard"].map((lvl) => (
                    <button
                      key={lvl}
                      type="button"
                      onClick={() => setDifficulty(lvl)}
                      className={cn(
                        "px-3 py-2 rounded-lg text-sm capitalize border transition-colors",
                        difficulty === lvl
                          ? "bg-emerald-600/20 border-emerald-600 text-emerald-200"
                          : "border-gray-700 text-gray-400 hover:border-gray-600"
                      )}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-gray-500 mb-1.5 block">
                  Number of Questions ({questionCount})
                </label>
                <input
                  type="range"
                  min={3}
                  max={15}
                  value={questionCount}
                  onChange={(e) => setQuestionCount(Number(e.target.value))}
                  className="w-full accent-emerald-500"
                />
              </div>
            </div>

            <div className="flex gap-3 justify-end">
              <button onClick={() => setShowModal(false)} className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200">
                Cancel
              </button>
              <button
                onClick={generateQuiz}
                disabled={!selectedDoc || generating}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm rounded-lg transition-colors"
              >
                {generating ? "Generating..." : "Generate"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
