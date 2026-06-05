"use client";
import { useEffect, useRef, useState } from "react";
import type { AnalyzeResponse, ChatMessage, Citation, VideoMetadata } from "./types";
import VideoCard from "./components/VideoCard";
import ChatPanel from "./components/ChatPanel";
import { SUGGESTED } from "./suggested";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

export default function HomePage() {
  const [urlA, setUrlA] = useState("");
  const [urlB, setUrlB] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeError, setAnalyzeError] = useState<string | null>(null);
  const [session, setSession] = useState<AnalyzeResponse | null>(null);
  const [backendOk, setBackendOk] = useState<boolean | null>(null);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const r = await fetch(`${API_BASE}/healthz`);
        const d = await r.json();
        if (!cancelled) {
          setBackendOk(Boolean(d?.status === "ok" && d?.openai_configured));
        }
      } catch {
        if (!cancelled) setBackendOk(false);
      }
    }
    check();
    const t = setInterval(check, 5000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  async function runAnalyze() {
    setAnalyzeError(null);
    if (!urlA.trim() || !urlB.trim()) {
      setAnalyzeError("Please provide both URLs.");
      return;
    }
    setAnalyzing(true);
    setSession(null);
    setMessages([]);
    try {
      const r = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ url_a: urlA.trim(), url_b: urlB.trim() }),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || `HTTP ${r.status}`);
      }
      const data: AnalyzeResponse = await r.json();
      setSession(data);
    } catch (e: any) {
      setAnalyzeError(e.message || "Failed to analyze videos");
    } finally {
      setAnalyzing(false);
    }
  }

  async function sendMessage(text: string) {
    if (!session) return;
    setChatError(null);
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: text,
    };
    const aiId = crypto.randomUUID();
    const placeholder: ChatMessage = {
      id: aiId,
      role: "assistant",
      content: "",
      citations: [],
      pending: true,
    };
    setMessages((m) => [...m, userMsg, placeholder]);
    setStreaming(true);

    try {
      const r = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ session_id: session.session_id, message: text }),
      });
      if (!r.ok || !r.body) {
        throw new Error(`Chat request failed: ${r.status}`);
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const collected: Citation[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames separated by blank lines
        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          const lines = frame.split("\n");
          let event = "message";
          let data = "";
          for (const ln of lines) {
            if (ln.startsWith("event:")) event = ln.slice(6).trim();
            else if (ln.startsWith("data:")) data += ln.slice(5).trim();
          }
          if (!data) continue;
          if (event === "token") {
            setMessages((m) =>
              m.map((msg) =>
                msg.id === aiId
                  ? { ...msg, content: msg.content + data }
                  : msg,
              ),
            );
          } else if (event === "citation") {
            try {
              const c = JSON.parse(data);
              collected.push(c);
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === aiId
                    ? { ...msg, citations: [...(msg.citations || []), c] }
                    : msg,
                ),
              );
            } catch {}
          } else if (event === "error") {
            setChatError(JSON.parse(data).message || "Stream error");
          } else if (event === "done") {
            setMessages((m) =>
              m.map((msg) => (msg.id === aiId ? { ...msg, pending: false } : msg)),
            );
          }
        }
      }
    } catch (e: any) {
      setChatError(e.message || "Chat failed");
      setMessages((m) =>
        m.map((msg) =>
          msg.id === aiId ? { ...msg, pending: false, content: msg.content || "(stream error)" } : msg,
        ),
      );
    } finally {
      setStreaming(false);
    }
  }

  return (
    <main className="mx-auto max-w-[1400px] px-4 py-6 lg:px-6">
      <header className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">
            <span className="gradient-text">VidCompare</span>{" "}
            <span className="text-zinc-400 font-normal">— RAG chat for two videos</span>
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Paste a YouTube link and an Instagram Reel. Get an AI analyst that cites the transcript.
          </p>
        </div>
        <a
          href="https://github.com/langchain-ai/langgraph"
          target="_blank"
          rel="noreferrer"
          className="hidden text-xs text-zinc-500 hover:text-zinc-300 md:block"
        >
          built with LangGraph · Chroma · OpenAI
        </a>
        <div className="ml-4 hidden items-center gap-2 text-xs md:flex">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              backendOk === null
                ? "bg-zinc-500"
                : backendOk
                ? "bg-emerald-400"
                : "bg-amber-400"
            }`}
          />
          <span className="text-zinc-400">
            {backendOk === null
              ? "checking backend…"
              : backendOk
              ? "backend ready"
              : "backend missing OPENAI_API_KEY"}
          </span>
        </div>
      </header>

      <section className="mb-6 rounded-2xl border border-ink-700 bg-ink-900/70 p-4 backdrop-blur">
        <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_1fr_auto]">
          <input
            placeholder="Video A — YouTube URL"
            value={urlA}
            onChange={(e) => setUrlA(e.target.value)}
            className="w-full rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-neon-500 focus:outline-none"
          />
          <input
            placeholder="Video B — Instagram Reel URL"
            value={urlB}
            onChange={(e) => setUrlB(e.target.value)}
            className="w-full rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-neon-500 focus:outline-none"
          />
          <button
            onClick={runAnalyze}
            disabled={analyzing}
            className="rounded-lg bg-gradient-to-r from-neon-500 to-blue-500 px-5 py-2 text-sm font-semibold text-white shadow-lg shadow-neon-500/20 transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {analyzing ? "Analyzing…" : "Analyze"}
          </button>
        </div>
        {analyzeError && (
          <p className="mt-2 text-sm text-red-400">{analyzeError}</p>
        )}
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.1fr_1fr]">
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <VideoCard
            label="A"
            accent="from-fuchsia-500 to-pink-500"
            meta={session?.video_a || null}
            loading={analyzing}
            placeholder="YouTube analysis will appear here"
          />
          <VideoCard
            label="B"
            accent="from-cyan-500 to-blue-500"
            meta={session?.video_b || null}
            loading={analyzing}
            placeholder="Instagram analysis will appear here"
          />
          {session && (
            <div className="col-span-1 sm:col-span-2 rounded-xl border border-ink-700 bg-ink-900/60 px-3 py-2 text-xs text-zinc-400">
              Indexed <span className="text-zinc-200 font-semibold">{session.chunks_indexed}</span>{" "}
              transcript chunks · session{" "}
              <code className="text-zinc-300">{session.session_id}</code>
            </div>
          )}
        </section>

        <section className="min-h-[520px]">
          <ChatPanel
            messages={messages}
            streaming={streaming}
            onSend={sendMessage}
            error={chatError}
            disabled={!session}
            suggestions={SUGGESTED}
          />
        </section>
      </div>
    </main>
  );
}
