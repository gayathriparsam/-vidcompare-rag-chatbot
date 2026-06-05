"use client";
import type { VideoMetadata } from "../types";

function fmt(n?: number | null) {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function formatDuration(s?: number | null) {
  if (!s) return "—";
  const m = Math.floor(s / 60);
  const ss = s % 60;
  return `${m}:${ss.toString().padStart(2, "0")}`;
}

function formatDate(d?: string | null) {
  if (!d) return "—";
  // yt-dlp returns YYYYMMDD
  if (/^\d{8}$/.test(d)) {
    return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
  }
  return d;
}

export default function VideoCard({
  label,
  accent,
  meta,
  loading,
  placeholder,
}: {
  label: string;
  accent: string;
  meta: VideoMetadata | null;
  loading: boolean;
  placeholder: string;
}) {
  return (
    <div className="overflow-hidden rounded-2xl border border-ink-700 bg-ink-900/70 backdrop-blur">
      <div className={`bg-gradient-to-r ${accent} px-4 py-2 text-sm font-semibold text-white`}>
        Video {label}
        <span className="ml-2 text-xs font-normal opacity-80">
          {meta ? `· ${meta.platform}` : ""}
        </span>
      </div>

      {loading && (
        <div className="p-4">
          <div className="aspect-video w-full animate-pulse rounded-lg bg-ink-800" />
          <div className="mt-3 h-4 w-3/4 animate-pulse rounded bg-ink-800" />
          <div className="mt-2 h-3 w-1/2 animate-pulse rounded bg-ink-800" />
        </div>
      )}

      {!loading && !meta && (
        <div className="p-6 text-sm text-zinc-500">{placeholder}</div>
      )}

      {!loading && meta && (
        <div className="p-4">
          <a href={meta.url} target="_blank" rel="noreferrer" className="block">
            {meta.thumbnail ? (
              <img
                src={meta.thumbnail}
                alt={meta.title}
                className="aspect-video w-full rounded-lg object-cover"
                loading="lazy"
              />
            ) : (
              <div className="flex aspect-video w-full items-center justify-center rounded-lg bg-ink-800 text-zinc-500">
                no thumbnail
              </div>
            )}
          </a>
          <h3 className="mt-3 line-clamp-2 text-sm font-semibold text-zinc-100">
            {meta.title}
          </h3>
          <p className="mt-1 text-xs text-zinc-400">
            {meta.creator}
            {meta.creator_followers ? (
              <span className="text-zinc-500"> · {fmt(meta.creator_followers)} followers</span>
            ) : null}
          </p>

          <div className="mt-3 grid grid-cols-3 gap-2 text-center">
            <Stat label="Views" value={fmt(meta.views)} />
            <Stat label="Likes" value={fmt(meta.likes)} />
            <Stat label="Comments" value={fmt(meta.comments)} />
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 text-center">
            <Stat
              label="Engagement"
              value={`${meta.engagement_rate.toFixed(2)}%`}
              highlight
            />
            <Stat label="Duration" value={formatDuration(meta.duration_seconds)} />
            <Stat label="Uploaded" value={formatDate(meta.upload_date)} />
          </div>

          {meta.hashtags?.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {meta.hashtags.slice(0, 8).map((h) => (
                <span
                  key={h}
                  className="rounded-full bg-ink-800 px-2 py-0.5 text-[10px] text-zinc-300"
                >
                  #{h}
                </span>
              ))}
              {meta.hashtags.length > 8 && (
                <span className="text-[10px] text-zinc-500">
                  +{meta.hashtags.length - 8}
                </span>
              )}
            </div>
          )}
          <p className="mt-2 text-[10px] text-zinc-500">
            transcript: {meta.transcript_chars.toLocaleString()} chars
          </p>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div
      className={`rounded-lg border px-2 py-1.5 ${
        highlight
          ? "border-neon-500/40 bg-neon-500/10"
          : "border-ink-700 bg-ink-800/70"
      }`}
    >
      <div className={`text-sm font-semibold ${highlight ? "text-neon-400" : "text-zinc-100"}`}>
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
    </div>
  );
}
