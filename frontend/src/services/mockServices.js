import { evaluations, samples } from '../data/mockData';

export const analysisService = {
  async getAnalysis(id = 'SAMPLE-001') {
    return samples.find((sample) => sample.id === id) ?? samples[0];
  },
  async startAnalysis() {
    return { runId: `run-${Date.now()}`, status: 'processing' };
  }
};

export const datasetService = {
  async listSamples() { return samples; },
  async getSample(id) { return samples.find((sample) => sample.id === id) ?? samples[0]; }
};

export const evaluationService = {
  async listEvaluations() { return evaluations; },
  async getEvaluation(index = 0) { return evaluations[index] ?? evaluations[0]; }
};
