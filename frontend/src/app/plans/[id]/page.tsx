"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { ArrowLeft, CheckCircle, Circle, Clock, ChevronRight } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { LearningPlanDetail, PlanTopic } from "@/lib/types";
import { cn } from "@/lib/utils";

const statusIcon = {
  not_started: <Circle className="w-4 h-4 text-gray-600" />,
  in_progress: <Clock className="w-4 h-4 text-blue-400" />,
  completed: <CheckCircle className="w-4 h-4 text-emerald-500" />,
};

function TopicItem({
  topic,
  planId,
  onStatusChange,
  depth = 0,
}: {
  topic: PlanTopic;
  planId: string;
  onStatusChange: (topicId: string, status: string) => void;
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
        >
          {statusIcon[topic.status]}
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className={cn("text-sm font-medium", topic.status === "completed" ? "line-through text-gray-500" : "text-gray-200")}>
              {topic.title}
            </p>
            {topic.estimated_hours && (
              <span className="text-xs text-gray-600 shrink-0">{topic.estimated_hours}h</span>
            )}
          </div>
          {topic.description && (
            <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">{topic.description}</p>
          )}
        </div>
        {topic.subtopics.length > 0 && (
          <button onClick={() => setExpanded(!expanded)} className="text-gray-600 hover:text-gray-400 shrink-0">
            <ChevronRight className={cn("w-4 h-4 transition-transform", expanded && "rotate-90")} />
          </button>
        )}
      </div>
      {expanded && topic.subtopics.length > 0 && (
        <div>
          {topic.subtopics.map((sub) => (
            <TopicItem key={sub.id} topic={sub} planId={planId} onStatusChange={onStatusChange} depth={depth + 1} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function PlanDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [plan, setPlan] = useState<LearningPlanDetail | null>(null);

  useEffect(() => {
    api.plans.get(id).then(setPlan);
  }, [id]);

  async function handleStatusChange(topicId: string, status: string) {
    if (!plan) return;
    await api.plans.updateTopic(id, topicId, status);
    // Reload to reflect changes
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

      {/* Topics */}
      <div className="space-y-1">
        {plan.topics.map((topic) => (
          <TopicItem key={topic.id} topic={topic} planId={id} onStatusChange={handleStatusChange} />
        ))}
      </div>
    </div>
  );
}
