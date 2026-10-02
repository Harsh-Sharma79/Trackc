export type FlawSeverity = 'low' | 'medium' | 'high' | 'critical' | string;

export interface AudioMetadata {
  name: string;
  duration_seconds: number;
  format?: string;
  sample_rate?: number;
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  confidence?: number;
}

export interface FlawRegion {
  id?: string;
  type: string;
  start: number;
  end: number;
  severity: FlawSeverity;
  explanation?: string;
  affected_text?: string;
  confidence?: number;
  feature?: string;
}

export interface RubricScore {
  name: string;
  score: number;
  explanation?: string;
  status?: string;
  weight?: number;
}

export interface EvaluationResult {
  id: string;
  status: 'complete' | 'processing' | 'error' | string;
  overall_score: number;
  summary?: string;
  duration_seconds: number;
  transcript: TranscriptWord[];
  flaws: FlawRegion[];
  rubric_scores: RubricScore[];
  participant_audio?: AudioMetadata;
  reference_audio?: AudioMetadata;
  evaluated_at?: string;
  notes?: string[];
  metadata?: Record<string, unknown>;
}

export interface CapabilitiesResponse {
  microphone?: boolean;
  reference_audio?: boolean;
  features?: string[];
  max_duration_seconds?: number;
}

export interface EvaluateRequest {
  transcript: string;
  participant_audio: File;
  reference_audio?: File;
}
