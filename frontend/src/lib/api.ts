const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}/api/v1${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

// Knowledge
export const api = {
  knowledge: {
    upload: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return request<import("./types").Document>("/knowledge/upload", {
        method: "POST",
        headers: {},
        body: form,
      });
    },
    list: () => request<import("./types").Document[]>("/knowledge/documents"),
    get: (id: string) => request<import("./types").Document>(`/knowledge/documents/${id}`),
    delete: (id: string) =>
      fetch(`${BASE_URL}/api/v1/knowledge/documents/${id}`, { method: "DELETE" }),
  },

  chat: {
    createSession: (title = "New Chat") =>
      request<import("./types").ChatSession>("/chat/sessions", {
        method: "POST",
        body: JSON.stringify({ title }),
      }),
    listSessions: () => request<import("./types").ChatSession[]>("/chat/sessions"),
    getSession: (id: string) =>
      request<import("./types").ChatSessionDetail>(`/chat/sessions/${id}`),
    deleteSession: (id: string) =>
      fetch(`${BASE_URL}/api/v1/chat/sessions/${id}`, { method: "DELETE" }),
    sendMessage: (sessionId: string, content: string): EventSource => {
      // We can't use EventSource with POST, so we use fetch + ReadableStream
      return null as unknown as EventSource;
    },
    streamMessage: (sessionId: string, content: string): Promise<Response> =>
      fetch(`${BASE_URL}/api/v1/chat/sessions/${sessionId}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content }),
      }),
  },

  plans: {
    generate: (goal: string, documentIds?: string[]) =>
      request<import("./types").LearningPlanDetail>("/plans/generate", {
        method: "POST",
        body: JSON.stringify({ goal, document_ids: documentIds?.length ? documentIds : null }),
      }),
    list: () => request<import("./types").LearningPlan[]>("/plans"),
    get: (id: string) => request<import("./types").LearningPlanDetail>(`/plans/${id}`),
    updateTopic: (planId: string, topicId: string, status: string) =>
      request<import("./types").PlanTopic>(`/plans/${planId}/topics/${topicId}`, {
        method: "PATCH",
        body: JSON.stringify({ status }),
      }),
    lesson: (planId: string, topicId: string, opts?: { language?: string; regenerate?: boolean }) =>
      request<import("./types").TopicLesson>(`/plans/${planId}/topics/${topicId}/lesson`, {
        method: "POST",
        body: JSON.stringify({ language: opts?.language ?? "auto", regenerate: opts?.regenerate ?? false }),
      }),
    gradeExercise: (planId: string, topicId: string, exercise: string, answer: string, language = "auto") =>
      request<import("./types").GradeResult>(`/plans/${planId}/topics/${topicId}/grade`, {
        method: "POST",
        body: JSON.stringify({ exercise, answer, language }),
      }),
    topicChat: (planId: string, topicId: string, message: string, history: Array<{ role: string; content: string }>): Promise<Response> =>
      fetch(`${BASE_URL}/api/v1/plans/${planId}/topics/${topicId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, history }),
      }),
  },

  quizzes: {
    generate: (params: { document_id?: string; topic_id?: string; question_count?: number; difficulty?: string }) =>
      request<import("./types").QuizDetail>("/quizzes/generate", {
        method: "POST",
        body: JSON.stringify(params),
      }),
    list: () => request<import("./types").Quiz[]>("/quizzes"),
    get: (id: string) => request<import("./types").QuizDetail>(`/quizzes/${id}`),
    submitAttempt: (quizId: string, answers: Array<{ question_id: string; answer: string }>) =>
      request("/quizzes/" + quizId + "/attempts", {
        method: "POST",
        body: JSON.stringify({ answers }),
      }),
  },

  progress: {
    summary: () => request<import("./types").ProgressSummary>("/progress/summary"),
    events: () => request<unknown[]>("/progress/events"),
  },
};
