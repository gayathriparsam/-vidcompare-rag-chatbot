"use client";
import { useEffect, useRef, useState } from "react";
import type { ChatMessage, Citation } from "../types";

export default function ChatPanel({
  messages,
  streaming,
  onSend,
  error,
  disabled,
  suggestions,
}: {
  messages: ChatMessage[];
  streaming: boolean;
  onSend: (text: string) => void;
  error: string | null;
  disabled: boolean;
  suggestions: string[];
}) {
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  function submit(e?: React.FormEvent) {
    e?.preventDefault();
    const t = draft.trim();
    if (!t || streaming) return;
    setDraft("");
    onSend(t);
  }

  return (
    <div className="flex h-full min-h-[520px] flex-col rounded-2xl border border-ink-700 bg-ink-900/70 backdrop-blur">
      <div className="border-b border-ink-700 px-4 py-3">
        <div className="text-sm font-semibold text-zinc-100">Analyst chat</div>
        <div className="text-xs text-zinc-500">
          Citations appear under each answer. Memory is kept for this session.
        </div>
      </div>

      <div ref={scrollRef} className="scrollbar-thin flex-1 space-y-3 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="rounded-xl border border-dashed border-ink-700 p-4 text-sm text-zinc-400">
            {disabled
              ? "Analyze two videos to start chatting."
              : "Ask anything about the two videos. Try a suggestion below."}
          </div>
        )}
        {messages.map((m) => (
          <Bubble key={m.id} message={m} />
        ))}
        {error && (
          <div className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      {messages.length === 0 && !disabled && (
        <div className="border-t border-ink-700 px-4 py-2">
          <div className="flex flex-wrap gap-1.5">
            {suggestions.map((s) => (
              <button
                key={s}
                onClick={() => onSend(s)}
                className="rounded-full border border-ink-700 bg-ink-800 px-3 py-1 text-xs text-zinc-200 hover:border-neon-500 hover:text-white"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}

      <form
        onSubmit={submit}
        className="flex gap-2 border-t border-ink-700 p-3"
      >
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={disabled || streaming}
          placeholder={
            disabled
              ? "Analyze two videos first…"
              : streaming
              ? "Streaming…"
              : "Ask a question about Video A vs B…"
          }
          className="flex-1 rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-neon-500 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={disabled || streaming || !draft.trim()}
          className="rounded-lg bg-gradient-to-r from-neon-500 to-blue-500 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </div>
  );
}

function Bubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[90%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-sm leading-relaxed ${
          isUser
            ? "bg-gradient-to-r from-neon-500 to-blue-500 text-white"
            : "border border-ink-700 bg-ink-800/70 text-zinc-100"
        }`}
      >
        <div className={message.pending && !isUser ? "streaming-cursor" : ""}>
          {message.content || (message.pending ? "…" : "")}
        </div>
        {!isUser && message.citations && message.citations.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {uniq(message.citations).map((c, i) => (
              <CitationChip key={i} c={c} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function CitationChip({ c }: { c: Citation }) {
  return (
    <span
      title={c.snippet}
      className="inline-flex items-center gap-1 rounded-full border border-ink-700 bg-ink-900 px-2 py-0.5 text-[10px] text-zinc-300"
    >
      <span
        className={`inline-block h-1.5 w-1.5 rounded-full ${
          c.video_id === "A" ? "bg-fuchsia-400" : "bg-cyan-400"
        }`}
      />
      {c.video_id}·{c.chunk_index}
    </span>
  );
}

function uniq(arr: Citation[]): Citation[] {
  const seen = new Set<string>();
  const out: Citation[] = [];
  for (const c of arr) {
    const k = `${c.video_id}-${c.chunk_index}`;
    if (!seen.has(k)) {
      seen.add(k);
      out.push(c);
    }
  }
  return out;
}
