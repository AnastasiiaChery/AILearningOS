import type { Citation } from "./types";

export interface StreamToken {
  type: "token";
  text: string;
}

export interface StreamDone {
  type: "done";
  citations: Citation[];
}

export interface StreamError {
  type: "error";
  text: string;
}

export type StreamEvent = StreamToken | StreamDone | StreamError;

export async function* readSSEStream(response: Response): AsyncGenerator<StreamEvent> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6).trim();
        if (!data) continue;
        try {
          const event = JSON.parse(data) as StreamEvent;
          yield event;
        } catch {
          // ignore malformed lines
        }
      }
    }
  }
}
