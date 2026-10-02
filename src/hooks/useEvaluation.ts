import { useCallback, useState } from 'react';
import { demoResult } from '../demo/demoResult';
import { evaluateSpeech } from '../lib/api';
import type { EvaluationResult, EvaluateRequest } from '../types/evaluation';

const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';

export function useEvaluation() {
  const [result, setResult] = useState<EvaluationResult | null>(DEMO_MODE ? demoResult : null);
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'complete' | 'error'>(DEMO_MODE ? 'complete' : 'idle');
  const [error, setError] = useState<string | null>(null);

  const runEvaluation = useCallback(async (request: EvaluateRequest) => {
    setStatus('analyzing');
    setError(null);
    try {
      const nextResult = DEMO_MODE ? demoResult : await evaluateSpeech(request);
      setResult(nextResult);
      setStatus('complete');
      return nextResult;
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : 'Analysis failed. Please retry.';
      setError(message);
      setStatus('error');
      throw cause;
    }
  }, []);

  return { result, status, error, runEvaluation, isDemo: DEMO_MODE };
}
