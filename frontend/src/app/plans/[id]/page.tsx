"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { ArrowLeft, CheckCircle, Circle, Clock, ChevronRight, BookOpen, X, Sparkles, RefreshCw, Send } from "lucide-react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "@/lib/api";
import { readSSEStream } from "@/lib/streaming";
import type { LearningPlanDetail, PlanTopic, TopicLesson, GradeResult } from "@/lib/types";
import { cn } from "@/lib/utils";

const statusIcon = {
  not_started: <Circle className="w-4 h-4 text-gray-600" />,
  in_progress: <Clock className="w-4 h-4 text-blue-400" />,
  completed: <CheckCircle className="w-4 h-4 text-emerald-500" />,
};

const verdictStyle = {
  correct: "bg-emerald-500/10 text-emerald-400",
  partial: "bg-amber-500/10 text-amber-400",
  incorrect: "bg-red-500/10 text-red-400",
};

function StudyModal({
  planId,
  topic,
  onClose,
  onComplete,
}: {
  planId: string;
  topic: PlanTopic;
  onClose: () => void;
  onComplete: () => void;
}) {
  const router = useRouter();
  const [lesson, setLesson] = useState<TopicLesson | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [language, setLanguage] = useState("auto");
  const [loading, setLoading] = useState(true);
  const [answer, setAnswer] = useState("");
  const [grading, setGrading] = useState(false);
  const [result, setResult] = useState<GradeResult | null>(null);
  const [makingQuiz, setMakingQuiz] = useState(false);

  const [tab, setTab] = useState<"lesson" | "ask">("lesson");
  const [chat, setChat] = useState<{ role: "user" | "assistant"; content: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatSending, setChatSending] = useState(false);

  async function sendChat() {
    const msg = chatInput.trim();
    if (!msg || chatSending) return;
    setChatInput("");
    setChatSending(true);
    const history = chat.map((m) => ({ role: m.role, content: m.content }));
    setChat((prev) => [...prev, { role: "user", content: msg }, { role: "assistant", content: "" }]);
    try {
      const res = await api.plans.topicChat(planId, topic.id, msg, history);
      let full = "";
      for await (const ev of readSSEStream(res)) {
        if (ev.type === "token") {
          full += ev.text;
          setChat((prev) => prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: full } : m)));
        }
      }
      if (!full) {
        setChat((prev) => prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: "Sorry, an error occurred." } : m)));
      }
    } catch {
      setChat((prev) => prev.map((m, i) => (i === prev.length - 1 ? { ...m, content: "Connection error." } : m)));
    } finally {
      setChatSending(false);
    }
  }

  const LANGS: { value: string; label: string }[] = [
    { value: "auto", label: "Auto (source)" },
    { value: "uk", label: "Українська" },
    { value: "en", label: "English" },
    { value: "ru", label: "Русский" },
  ];

  function loadLesson(lang: string, regenerate = false) {
    setLoading(true);
    setError(null);
    setLesson(null);
    setResult(null);
    setAnswer("");
    api.plans
      .lesson(planId, topic.id, { language: lang, regenerate })
      .then(setLesson)
      .catch((e) => setError((e as Error).message || "Failed to load lesson"))
      .finally(() => setLoading(false));
  }

  useEffect(() => {
    loadLesson(language);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [planId, topic.id, language]);

  async function submitAnswer() {
    if (!lesson || !answer.trim()) return;
    setGrading(true);
    try {
      setResult(await api.plans.gradeExercise(planId, topic.id, lesson.exercise, answer.trim(), language));
    } catch (e: unknown) {
      setResult({ verdict: "partial", explanation: (e as Error).message || "Grading failed." });
    } finally {
      setGrading(false);
    }
  }

  async function fullQuiz() {
    setMakingQuiz(true);
    try {
      const quiz = await api.quizzes.generate({ topic_id: topic.id });
      router.push(`/quizzes/${quiz.id}`);
    } catch (e: unknown) {
      alert((e as Error).message || "Failed to generate quiz");
      setMakingQuiz(false);
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border border-gray-800 rounded-2xl w-full max-w-2xl max-h-[88vh] flex flex-col">
        <div className="flex items-center justify-between p-5 border-b border-gray-800 shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <BookOpen className="w-4 h-4 text-emerald-400 shrink-0" />
            <h2 className="text-base font-semibold text-white truncate">{topic.title}</h2>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              title="Lesson language"
              className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1 text-xs text-gray-200 outline-none focus:border-emerald-500"
            >
              {LANGS.map((l) => (
                <option key={l.value} value={l.value}>{l.label}</option>
              ))}
            </select>
            <button
              onClick={() => loadLesson(language, true)}
              disabled={loading}
              title="Regenerate lesson"
              className="p-1.5 text-gray-500 hover:text-emerald-400 disabled:opacity-40 transition-colors"
            >
              <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
            </button>
            <button onClick={onClose} className="text-gray-500 hover:text-gray-200">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-5 pt-3 shrink-0">
          {(["lesson", "ask"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                "px-3 py-1.5 text-sm rounded-lg transition-colors",
                tab === t ? "bg-gray-800 text-emerald-300" : "text-gray-500 hover:text-gray-300"
              )}
            >
              {t === "lesson" ? "Lesson & practice" : "Ask about this"}
            </button>
          ))}
        </div>

        <div className="overflow-y-auto p-5 space-y-5">
          {/* Lesson */}
          {tab === "lesson" && (error ? (
            <p className="text-sm text-red-400">{error}</p>
          ) : !lesson ? (
            <p className="text-sm text-gray-500">Generating lesson from your knowledge base…</p>
          ) : (
            <>
              <div className="prose-chat text-gray-300">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{lesson.lesson}</ReactMarkdown>
              </div>

              {lesson.citations.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {lesson.citations.map((c, i) => (
                    <span
                      key={c.chunk_id + i}
                      title={c.heading || c.filename}
                      className="text-[11px] px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 border border-gray-700 max-w-[240px] truncate"
                    >
                      [{i + 1}] {c.filename}
                    </span>
                  ))}
                </div>
              )}

              {/* Exercise */}
              <div className="border-t border-gray-800 pt-4">
                <p className="text-xs font-semibold text-emerald-400 uppercase tracking-wide mb-2">
                  Practice
                </p>
                <p className="text-sm text-gray-200 mb-3">{lesson.exercise}</p>
                <textarea
                  value={answer}
                  onChange={(e) => setAnswer(e.target.value)}
                  disabled={!!result}
                  rows={4}
                  placeholder="Answer in your own words…"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-600 outline-none focus:border-emerald-500 resize-none"
                />

                {result && (
                  <div className="mt-3 space-y-2">
                    <span className={cn("inline-block px-2 py-0.5 rounded-full text-xs font-medium capitalize", verdictStyle[result.verdict])}>
                      {result.verdict}
                    </span>
                    <p className="text-sm text-gray-400">{result.explanation}</p>
                  </div>
                )}

                <div className="flex flex-wrap gap-2 mt-3">
                  {!result ? (
                    <button
                      onClick={submitAnswer}
                      disabled={!answer.trim() || grading}
                      className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm rounded-lg transition-colors"
                    >
                      {grading ? "Checking…" : "Check answer"}
                    </button>
                  ) : (
                    <button
                      onClick={() => { setResult(null); setAnswer(""); }}
                      className="px-4 py-2 border border-gray-700 text-gray-300 hover:border-gray-600 text-sm rounded-lg transition-colors"
                    >
                      Try again
                    </button>
                  )}
                  <button
                    onClick={fullQuiz}
                    disabled={makingQuiz}
                    className="px-4 py-2 border border-gray-700 text-gray-300 hover:border-gray-600 text-sm rounded-lg transition-colors flex items-center gap-1.5"
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    {makingQuiz ? "Building…" : "Full quiz on this topic"}
                  </button>
                  {topic.status !== "completed" && (
                    <button
                      onClick={() => { onComplete(); onClose(); }}
                      className="px-4 py-2 text-sm text-gray-400 hover:text-emerald-400 transition-colors ml-auto"
                    >
                      Mark complete
                    </button>
                  )}
                </div>
              </div>
            </>
          ))}

          {tab === "ask" && (
            <div className="flex flex-col h-[58vh]">
              <div className="flex-1 overflow-y-auto space-y-3 pr-1">
                {chat.length === 0 && (
                  <p className="text-sm text-gray-500">
                    Ask anything about “{topic.title}”. Answers use this topic&apos;s documents{lesson ? " and the lesson" : ""}.
                  </p>
                )}
                {chat.map((m, i) => (
                  <div key={i} className="text-sm">
                    <span className={cn("text-[10px] uppercase tracking-wide mr-2", m.role === "user" ? "text-gray-500" : "text-emerald-500")}>
                      {m.role === "user" ? "You" : "Mentor"}
                    </span>
                    {m.role === "assistant" ? (
                      <div className="prose-chat text-gray-300">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content || "…"}</ReactMarkdown>
                      </div>
                    ) : (
                      <span className="text-gray-200 whitespace-pre-wrap">{m.content}</span>
                    )}
                  </div>
                ))}
              </div>
              <div className="flex gap-2 pt-3 border-t border-gray-800 mt-3 shrink-0">
                <textarea
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); } }}
                  rows={1}
                  placeholder="Ask about this topic…"
                  className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-600 outline-none focus:border-emerald-500 resize-none"
                />
                <button
                  onClick={sendChat}
                  disabled={!chatInput.trim() || chatSending}
                  className="px-3 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded-lg transition-colors"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function TopicItem({
  topic,
  planId,
  onStatusChange,
  onStudy,
  depth = 0,
}: {
  topic: PlanTopic;
  planId: string;
  onStatusChange: (topicId: string, status: string) => void;
  onStudy: (topic: PlanTopic) => void;
  depth?: number;
}) {
  const [expanded, setExpanded] = useState(depth === 0);
  const nextStatus = {
    not_started: "in_progress",
    in_progress: "completed",
    completed: "not_started",
  }[topic.status];

  return (
    <div className={cn("border-l-2 pl-3", depth === 0 ? "border-gray-800" : "border-gray-700/50 ml-4")}>
      <div className="flex items-start gap-3 py-2.5">
        <button
          onClick={() => onStatusChange(topic.id, nextStatus)}
          className="mt-0.5 shrink-0 hover:scale-110 transition-transform"
          title="Toggle status"
        >
          {statusIcon[topic.status]}
        </button>
        <button onClick={() => onStudy(topic)} className="flex-1 min-w-0 text-left group">
          <div className="flex items-center gap-2">
            <p className={cn("text-sm font-medium group-hover:text-emerald-400 transition-colors", topic.status === "completed" ? "line-through text-gray-500" : "text-gray-200")}>
              {topic.title}
            </p>
            <BookOpen className="w-3.5 h-3.5 text-gray-600 group-hover:text-emerald-400 transition-colors shrink-0" />
            {topic.estimated_hours && (
              <span className="text-xs text-gray-600 shrink-0">{topic.estimated_hours}h</span>
            )}
          </div>
          {topic.description && (
            <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{topic.description}</p>
          )}
        </button>
        {topic.subtopics.length > 0 && (
          <button onClick={() => setExpanded(!expanded)} className="text-gray-600 hover:text-gray-400 shrink-0">
            <ChevronRight className={cn("w-4 h-4 transition-transform", expanded && "rotate-90")} />
          </button>
        )}
      </div>
      {expanded && topic.subtopics.length > 0 && (
        <div>
          {topic.subtopics.map((sub) => (
            <TopicItem key={sub.id} topic={sub} planId={planId} onStatusChange={onStatusChange} onStudy={onStudy} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function PlanDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [plan, setPlan] = useState<LearningPlanDetail | null>(null);
  const [studyTopic, setStudyTopic] = useState<PlanTopic | null>(null);

  useEffect(() => {
    api.plans.get(id).then(setPlan);
  }, [id]);

  async function handleStatusChange(topicId: string, status: string) {
    if (!plan) return;
    await api.plans.updateTopic(id, topicId, status);
    const updated = await api.plans.get(id);
    setPlan(updated);
  }

  if (!plan) return <div className="p-6 text-gray-500 text-sm">Loading...</div>;

  const allTopics = plan.topics.flatMap((t) => [t, ...t.subtopics]);
  const completed = allTopics.filter((t) => t.status === "completed").length;
  const total = allTopics.length;
  const progress = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center gap-3 mb-6">
        <Link href="/plans" className="p-1.5 text-gray-500 hover:text-gray-200 rounded transition-colors">
          <ArrowLeft className="w-4 h-4" />
        </Link>
        <h1 className="text-lg font-semibold text-white truncate">{plan.title}</h1>
      </div>

      {/* Progress bar */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">Progress</span>
          <span className="text-sm font-semibold text-emerald-400">{progress}%</span>
        </div>
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 rounded-full transition-all duration-500"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-xs text-gray-600 mt-2">{completed} / {total} topics completed</p>
      </div>

      {plan.description && (
        <p className="text-sm text-gray-500 mb-4">{plan.description}</p>
      )}

      <p className="text-xs text-gray-600 mb-2">Tap a topic to study it and practice.</p>

      {/* Topics */}
      <div className="space-y-1">
        {plan.topics.map((topic) => (
          <TopicItem key={topic.id} topic={topic} planId={id} onStatusChange={handleStatusChange} onStudy={setStudyTopic} />
        ))}
      </div>

      {studyTopic && (
        <StudyModal
          planId={id}
          topic={studyTopic}
          onClose={() => setStudyTopic(null)}
          onComplete={() => handleStatusChange(studyTopic.id, "completed")}
        />
      )}
    </div>
  );
}
