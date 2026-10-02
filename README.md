# Contrastive Speech Analytics

Frontend prototype for Track C: Contrastive Speech Analytics & Temporal Flaw Grounding.

## Run locally

```bash
npm install
npm run dev
```

The app supports an explicit frontend demo mode and a real Speech Evaluation API mode. Demo mode is enabled in `.env.example` so the interface can be explored without a backend. It is clearly labeled in the UI and never silently replaces a configured API in production.

For the real backend, create a `.env` file:

```bash
VITE_DEMO_MODE=false
VITE_API_URL=http://localhost:8000
```

The typed API adapter calls `POST /api/evaluate` with multipart audio/transcript data and `GET /api/capabilities`. The expected response contract lives in `src/types/evaluation.ts`.

## Routes

- `/` overview with interactive analysis demonstration
- `/analyze` primary analysis workspace
- `/dataset` contrastive dataset explorer and flaw spectrum
- `/evaluations` previous analysis runs
- `/methodology` technical pipeline
- `/settings` configurable mock rubric and display settings
- `/terms` prototype terms
- `/privacy` prototype privacy policy

## Validation

```bash
npm run build
```

## Architecture

- `src/data/mockData.js` contains the existing interactive dashboard fixture.
- `src/demo/demoResult.ts` contains a clearly marked `EvaluationResult` demo fixture.
- `src/lib/api.ts` is the dedicated real API client.
- `src/hooks/useEvaluation.ts` switches between explicit demo mode and the real API.
- `src/types/evaluation.ts` defines the backend-facing contract.
- `src/services/mockServices.js` remains the legacy dataset/history service boundary.
- `src/main.jsx` contains the route shell and reusable analytical UI components.
- `src/styles.css` contains the dark technical design system and responsive layout.
