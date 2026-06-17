"use client";

import { useEffect, useState } from "react";
import { BookOpen, Brain, CheckCircle, FileText, LayoutDashboard, Target } from "lucide-react";
import { api } from "@/lib/api";
import type { ProgressSummary } from "@/lib/types";

function StatCard({ icon: Icon, label, value, sub }: { icon: React.ElementType; label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-3 mb-2">
        <div className="p-2 bg-emerald-500/10 rounded-lg">
          <Icon className="w-4 h-4 text-emerald-400" />
        </div>
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
      {sub && <p className="text-xs text-gray-600 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function ProgressPage() {
  const [summary, setSummary] = useState<ProgressSummary | null>(null);

  useEffect(() => {
    api.progress.summary().then(setSummary);
  }, []);

  if (!summary) return <div className="p-6 text-gray-500 text-sm">Loading...</div>;

  const topicsProgress =
    summary.total_topics > 0
      ? Math.round((summary.completed_topics / summary.total_topics) * 100)
      : 0;

  return (
    <div className="flex-1 flex flex-col p-6 overflow-y-auto">
      <h1 className="text-xl font-semibold text-white mb-6">Learning Progress</h1>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <StatCard icon={FileText} label="Documents" value={summary.total_documents} sub={`${summary.total_chunks} chunks indexed`} />
        <StatCard icon={Brain} label="Learning Plans" value={summary.total_plans} />
        <StatCard
          icon={CheckCircle}
          label="Topics Completed"
          value={summary.completed_topics}
          sub={`of ${summary.total_topics} total`}
        />
        <StatCard
          icon={Target}
          label="Avg Quiz Score"
          value={summary.avg_quiz_score != null ? `${Math.round(summary.avg_quiz_score * 100)}%` : "—"}
          sub={`${summary.total_quiz_attempts} attempts`}
        />
      </div>

      {/* Topic progress bar */}
      {summary.total_topics > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Overall Learning Progress</span>
            <span className="text-sm font-semibold text-emerald-400">{topicsProgress}%</span>
          </div>
          <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 rounded-full transition-all duration-700"
              style={{ width: `${topicsProgress}%` }}
            />
          </div>
        </div>
      )}

      {summary.total_documents === 0 && (
        <div className="mt-8 text-center text-gray-600 text-sm">
          <LayoutDashboard className="w-10 h-10 mx-auto mb-3 text-gray-800" />
          <p>Upload documents and start learning to see your progress here.</p>
        </div>
      )}
    </div>
  );
}
