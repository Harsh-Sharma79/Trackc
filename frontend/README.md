# Contrastive Speech Analytics — Frontend

React/Vite interface for the full-stack speech evaluation project. The **Run Evaluation** route uploads audio and transcript data to the FastAPI backend. The Overview, Analyze Demo, Dataset, Evaluation History and Methodology views still use clearly marked frontend fixtures.

## Run locally

From the project root, install backend requirements as documented in the root README, then:

```bash
npm --prefix frontend install
# In another terminal, start FastAPI on port 8000 (see the root README)
npm --prefix frontend run dev
```

Vite proxies `/api` to `http://localhost:8000`. For a backend on another origin, create `frontend/.env.local` and set `VITE_API_URL=http://your-api-host:8000`.

Demo mode is opt-in with `VITE_DEMO_MODE=true`; by default the real API client is used.

## API contract

- `GET /api/capabilities`
- `POST /api/evaluate` with multipart fields `participant_audio`, optional `reference_audio`, and `transcript`

The client is in `src/lib/api.ts`, hook in `src/hooks/useEvaluation.ts`, and response types in `src/types/evaluation.ts`.

The overview includes a narrated walkthrough at `public/how-to-use.mp4`; the narration script is documented at `docs/how-to-use-narration.md`.

## Build

```bash
npm --prefix frontend run build
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
