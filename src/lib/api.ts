import type { CapabilitiesResponse, EvaluateRequest, EvaluationResult } from '../types/evaluation';

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

async function readJson<T>(response: Response): Promise<T> {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const message = body && typeof body.detail === 'string' ? body.detail : 'The analysis service is unavailable.';
    throw new ApiError(response.status, message);
  }
  return body as T;
}

export async function getCapabilities(): Promise<CapabilitiesResponse> {
  const response = await fetch(`${API_URL}/api/capabilities`);
  return readJson<CapabilitiesResponse>(response);
}

export async function evaluateSpeech(request: EvaluateRequest): Promise<EvaluationResult> {
  const form = new FormData();
  form.append('participant_audio', request.participant_audio);
  if (request.reference_audio) form.append('reference_audio', request.reference_audio);
  form.append('transcript', request.transcript);

  const response = await fetch(`${API_URL}/api/evaluate`, { method: 'POST', body: form });
  return readJson<EvaluationResult>(response);
}

export const apiConfig = { baseUrl: API_URL };
