"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Brain, Plus, X } from "lucide-react";
import { api } from "@/lib/api";
import type { Document, LearningPlan } from "@/lib/types";
import { formatDate, cn } from "@/lib/utils";

export default function PlansPage() {
  const router = useRouter();
  const [plans, setPlans] = useState<LearningPlan[]>([]);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [goal, setGoal] = useState("");
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    api.plans.list().then(setPlans);
    api.knowledge.list().then((docs) => setDocuments(docs.filter((d) => d.status === "ready")));
  }, []);

  function toggleDoc(id: string) {
    setSelectedDocs((prev) => (prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]));
  }

  async function generatePlan() {
    if (!goal.trim()) return;
    setGenerating(true);
    try {
      const plan = await api.plans.generate(goal.trim(), selectedDocs);
      router.push(`/plans/${plan.id}`);
    } catch (e: unknown) {
      alert((e as Error).message || "Failed to generate plan");
      setGenerating(false);
    }
  }

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-semibold text-white">Learning Plans</h1>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded-lg transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Plan
        </button>
      </div>

      {plans.length === 0 ? (
        <div className="flex flex-col items-center justify-center flex-1 text-center gap-4">
          <Brain className="w-12 h-12 text-gray-700" />
          <div>
            <p className="text-gray-400 font-medium">No learning plans yet</p>
            <p className="text-gray-600 text-sm mt-1">
              Describe your learning goal and the AI will create a personalized plan
            </p>
          </div>
          <button
            onClick={() => setShowModal(true)}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm rounded-lg transition-colors"
          >
            Create First Plan
          </button>
        </div>
      ) : (
        <div className="grid gap-3">
          {plans.map((plan) => (
            <a
              key={plan.id}
              href={`/plans/${plan.id}`}
              className="block p-4 bg-gray-900 border border-gray-800 rounded-xl hover:border-gray-700 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-200">{plan.title}</p>
                  {plan.description && (
                    <p className="text-xs text-gray-500 mt-1 line-clamp-2">{plan.description}</p>
                  )}
                  <p className="text-xs text-gray-600 mt-2">Goal: {plan.goal}</p>
                </div>
                <span className="text-xs text-gray-600 shrink-0 ml-4">{formatDate(plan.created_at)}</span>
              </div>
            </a>
          ))}
        </div>
      )}

      {/* Generate Plan Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 w-full max-w-md">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-white">Create Learning Plan</h2>
              <button onClick={() => setShowModal(false)} className="text-gray-500 hover:text-gray-200">
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-sm text-gray-500 mb-4">
              Describe what you want to learn. The AI will analyze your knowledge base and create a personalized plan.
            </p>
            <textarea
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="e.g. I want to master FastAPI and build production-ready REST APIs"
              rows={4}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 placeholder-gray-600 outline-none focus:border-emerald-500 resize-none mb-4"
            />

            {documents.length > 0 && (
              <div className="mb-4">
                <label className="text-xs text-gray-500 mb-1.5 block">
                  Base on documents{" "}
                  <span className="text-gray-600">
                    {selectedDocs.length ? `(${selectedDocs.length} selected)` : "(all, if none selected)"}
                  </span>
                </label>
                <div className="max-h-40 overflow-y-auto rounded-lg border border-gray-800 divide-y divide-gray-800">
                  {documents.map((doc) => (
                    <button
                      key={doc.id}
                      type="button"
                      onClick={() => toggleDoc(doc.id)}
                      className={cn(
                        "w-full flex items-center gap-2 px-3 py-2 text-left text-sm transition-colors",
                        selectedDocs.includes(doc.id)
                          ? "bg-emerald-600/15 text-emerald-200"
                          : "text-gray-400 hover:bg-gray-800"
                      )}
                    >
                      <span
                        className={cn(
                          "w-4 h-4 rounded border flex items-center justify-center shrink-0 text-[10px]",
                          selectedDocs.includes(doc.id)
                            ? "bg-emerald-500 border-emerald-500 text-white"
                            : "border-gray-600"
                        )}
                      >
                        {selectedDocs.includes(doc.id) ? "✓" : ""}
                      </span>
                      <span className="truncate">{doc.original_filename}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setShowModal(false)}
                className="px-4 py-2 text-sm text-gray-400 hover:text-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={generatePlan}
                disabled={!goal.trim() || generating}
                className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-800 disabled:text-gray-600 text-white text-sm rounded-lg transition-colors"
              >
                {generating ? "Generating..." : "Generate Plan"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
