"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowLeft, CheckCircle, XCircle } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { QuizDetail } from "@/lib/types";
import { cn } from "@/lib/utils";

export default function QuizPage() {
  const { id } = useParams<{ id: string }>();
  const [quiz, setQuiz] = useState<QuizDetail | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [result, setResult] = useState<{ score: number; answers: Array<{ question_id: string; correct: boolean; correct_answer: string; explanation?: string; verdict?: "correct" | "partial" | "incorrect"; given_answer?: string }> } | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    api.quizzes.get(id).then(setQuiz);
  }, [id]);

  async function submitQuiz() {
    if (!quiz) return;
    setSubmitting(true);
    try {
      const payload = quiz.questions.map((q) => ({ question_id: q.id, answer: answers[q.id] || "" }));
      const res = await api.quizzes.submitAttempt(id, payload) as typeof result;
      setResult(res);
    } catch (e: unknown) {
      alert((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  if (!quiz) return <div className="p-6 text-gray-500 text-sm">Loading...</div>;

  const allAnswered = quiz.questions.every((q) => answers[q.id]);

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center gap-3 mb-6">
        <Link href="/quizzes" className="p-1.5 text-gray-500 hover:text-gray-200 rounded transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <h1 className="text-lg font-semibold text-white truncate">{quiz.title}</h1>
      </div>

      {/* Score Banner */}
      {result && (
        <div className={cn("rounded-xl p-4 mb-6 flex items-center gap-3", result.score >= 0.7 ? "bg-emerald-500/10 border border-emerald-500/20" : "bg-red-500/10 border border-red-500/20")}>
          {result.score >= 0.7
            ? <CheckCircle className="w-6 h-6 text-emerald-400" />
            : <XCircle className="w-6 h-6 text-red-400" />}
          <div>
            <p className={cn("font-semibold", result.score >= 0.7 ? "text-emerald-400" : "text-red-400")}>
              {Math.round(result.score * 100)}% — {result.score >= 0.7 ? "Great job!" : "Keep practicing!"}
            </p>
            <p className="text-xs text-gray-500">
              {result.answers.filter((a) => a.correct).length} / {quiz.questions.length} correct
            </p>
          </div>
        </div>
      )}

      <div className="space-y-6">
        {quiz.questions.map((q, idx) => {
          const resultForQ = result?.answers.find((a) => a.question_id === q.id);
          return (
            <div key={q.id} className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-sm font-medium text-gray-200 mb-3">
                {idx + 1}. {q.question_text}
              </p>

              {q.question_type === "short_answer" ? (
                <div className="space-y-2">
                  <input
                    type="text"
                    value={answers[q.id] || ""}
                    onChange={(e) => setAnswers((prev) => ({ ...prev, [q.id]: e.target.value }))}
                    disabled={!!result}
                    placeholder="Your answer..."
                    className={cn(
                      "w-full bg-gray-800 border rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-600 outline-none focus:border-emerald-500",
                      result && resultForQ?.verdict === "correct"
                        ? "border-emerald-500/40"
                        : result && resultForQ?.verdict === "partial"
                        ? "border-amber-500/40"
                        : result
                        ? "border-red-500/40"
                        : "border-gray-700"
                    )}
                  />
                  {result && resultForQ?.verdict && (
                    <div className="flex items-center gap-2 text-xs">
                      <span
                        className={cn(
                          "px-2 py-0.5 rounded-full font-medium capitalize",
                          resultForQ.verdict === "correct"
                            ? "bg-emerald-500/10 text-emerald-400"
                            : resultForQ.verdict === "partial"
                            ? "bg-amber-500/10 text-amber-400"
                            : "bg-red-500/10 text-red-400"
                        )}
                      >
                        {resultForQ.verdict}
                      </span>
                      {resultForQ.verdict !== "correct" && (
                        <span className="text-gray-500">
                          Expected: <span className="text-gray-300">{resultForQ.correct_answer}</span>
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  {Object.entries(q.options || {}).map(([key, text]) => {
                    const selected = answers[q.id] === key;
                    const isCorrect = resultForQ?.correct_answer === key;
                    const isWrong = result && selected && !isCorrect;
                    return (
                      <button
                        key={key}
                        onClick={() => !result && setAnswers((prev) => ({ ...prev, [q.id]: key }))}
                        disabled={!!result}
                        className={cn(
                          "w-full text-left px-3 py-2 rounded-lg text-sm border transition-colors",
                          result && isCorrect
                            ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-300"
                            : isWrong
                            ? "bg-red-500/10 border-red-500/30 text-red-300"
                            : selected
                            ? "bg-emerald-600/20 border-emerald-600 text-gray-100"
                            : "border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-200"
                        )}
                      >
                        <span className="font-mono text-xs mr-2 text-gray-500">{key}.</span>
                        {text}
                      </button>
                    );
                  })}
                </div>
              )}

              {result && resultForQ?.explanation && (
                <p className="text-xs text-gray-500 mt-3 border-t border-gray-800 pt-3">
                  💡 {resultForQ.explanation}
                </p>
              )}
            </div>
          );
        })}
      </div>

      {!result && (
        <button
          onClick={submitQuiz}
          disabled={!allAnswered || submitting}
          className="mt-6 px-6 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm font-medium rounded-xl transition-colors w-full"
        >
          {submitting ? "Submitting..." : "Submit Quiz"}
        </button>
      )}
    </div>
  );
}
