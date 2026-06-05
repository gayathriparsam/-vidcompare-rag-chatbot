export type VideoMetadata = {
  video_id: "A" | "B" | string;
  platform: "youtube" | "instagram" | string;
  url: string;
  title: string;
  creator: string;
  creator_followers?: number | null;
  views: number;
  likes: number;
  comments: number;
  hashtags: string[];
  upload_date?: string | null;
  duration_seconds?: number | null;
  thumbnail?: string | null;
  transcript_chars: number;
  engagement_rate: number;
};

export type Citation = {
  video_id: string;
  chunk_index: number;
  snippet: string;
};

export type AnalyzeResponse = {
  session_id: string;
  video_a: VideoMetadata;
  video_b: VideoMetadata;
  chunks_indexed: number;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  pending?: boolean;
};
