"use client";
import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

type Props = {
  token: string | null;
  email: string | null;
  onAuth: (token: string, email: string) => void;
  onLogout: () => void;
};

export default function LoginPanel({ token, email, onAuth, onLogout }: Props) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [formEmail, setFormEmail] = useState("");
  const [formPassword, setFormPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) {
      setError(null);
      setFormEmail("");
      setFormPassword("");
    }
  }, [open]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const path = mode === "login" ? "/api/auth/login" : "/api/auth/signup";
      const r = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email: formEmail.trim(), password: formPassword }),
      });
      if (!r.ok) {
        const txt = await r.text();
        try {
          const j = JSON.parse(txt);
          setError(j.detail || j.message || txt || `HTTP ${r.status}`);
        } catch {
          setError(txt || `HTTP ${r.status}`);
        }
        return;
      }
      const data = await r.json();
      onAuth(data.token, data.email);
      setOpen(false);
    } catch (err: any) {
      setError(err.message || "Network error");
    } finally {
      setSubmitting(false);
    }
  }

  if (token && email) {
    return (
      <div className="flex items-center gap-2 text-xs">
        <span className="hidden text-zinc-400 md:inline">signed in as</span>
        <span className="rounded-md bg-emerald-500/10 px-2 py-1 text-emerald-300 font-medium">
          {email}
        </span>
        <button
          onClick={onLogout}
          className="rounded-md border border-ink-700 bg-ink-800 px-2 py-1 text-zinc-300 transition hover:bg-ink-700"
        >
          log out
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="rounded-md border border-ink-700 bg-ink-800 px-3 py-1.5 text-xs text-zinc-200 transition hover:bg-ink-700"
      >
        {open ? "close" : "sign in"}
      </button>
      {open && (
        <div className="absolute right-0 top-full z-20 mt-2 w-80 rounded-xl border border-ink-700 bg-ink-900 p-4 shadow-2xl">
          <div className="mb-3 flex gap-1 rounded-lg border border-ink-700 bg-ink-800 p-1 text-xs">
            <button
              onClick={() => setMode("login")}
              className={`flex-1 rounded-md py-1.5 transition ${
                mode === "login" ? "bg-neon-500/20 text-neon-300" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Log in
            </button>
            <button
              onClick={() => setMode("signup")}
              className={`flex-1 rounded-md py-1.5 transition ${
                mode === "signup" ? "bg-neon-500/20 text-neon-300" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Sign up
            </button>
          </div>
          <form onSubmit={submit} className="space-y-2">
            <input
              type="email"
              required
              placeholder="email"
              value={formEmail}
              onChange={(e) => setFormEmail(e.target.value)}
              className="w-full rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-neon-500 focus:outline-none"
            />
            <input
              type="password"
              required
              minLength={8}
              placeholder="password (min 8 chars)"
              value={formPassword}
              onChange={(e) => setFormPassword(e.target.value)}
              className="w-full rounded-lg border border-ink-700 bg-ink-800 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-500 focus:border-neon-500 focus:outline-none"
            />
            {error && <p className="text-xs text-red-400">{error}</p>}
            <button
              type="submit"
              disabled={submitting}
              className="w-full rounded-lg bg-gradient-to-r from-neon-500 to-blue-500 px-3 py-2 text-sm font-semibold text-white transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {submitting
                ? "…"
                : mode === "login"
                ? "Log in"
                : "Create account"}
            </button>
          </form>
          <p className="mt-3 text-[10px] leading-relaxed text-zinc-500">
            {mode === "login"
              ? "Optional. Signing in lets you keep a list of past sessions."
              : "No password recovery in this demo. Pick something memorable."}
          </p>
        </div>
      )}
    </div>
  );
}
