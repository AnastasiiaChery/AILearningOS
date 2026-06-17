export interface Document {
  id: string;
  filename: string;
  original_filename: string;
  file_type: string;
  file_size: number;
  status: "pending" | "processing" | "ready" | "error";
  chunk_count: number | null;
  error_msg: string | null;
  created_at: string;
  updated_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Citation {
  chunk_id: string;
  filename: string;
  heading: string | null;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant";
  content: string;
  source_chunks: string[] | null;
  created_at: string;
  citations?: Citation[];
}

export interface ChatSessionDetail extends ChatSession {
  messages: ChatMessage[];
}

export interface PlanTopic {
  id: string;
  plan_id: string;
  parent_id: string | null;
  title: string;
  description: string | null;
  order_index: number;
  status: "not_started" | "in_progress" | "completed";
  estimated_hours: number | null;
  subtopics: PlanTopic[];
}

export interface LearningPlan {
  id: string;
  title: string;
  description: string | null;
  goal: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface LearningPlanDetail extends LearningPlan {
  topics: PlanTopic[];
}

export interface QuizQuestion {
  id: string;
  question_text: string;
  question_type: string;
  options: Record<string, string> | null;
  order_index: number;
}

export interface Quiz {
  id: string;
  title: string;
  topic_id: string | null;
  document_id: string | null;
  created_at: string;
}

export interface QuizDetail extends Quiz {
  questions: QuizQuestion[];
}

export interface ProgressSummary {
  total_documents: number;
  total_chunks: number;
  total_plans: number;
  completed_topics: number;
  total_topics: number;
  total_quiz_attempts: number;
  avg_quiz_score: number | null;
}
