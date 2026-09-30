# Contrastive Speech Analytics

Frontend prototype for Track C: Contrastive Speech Analytics & Temporal Flaw Grounding.

## Run locally

```bash
npm install
npm run dev
```

The app uses deterministic mock data and a mock service layer so a real audio-analysis backend can be connected later without redesigning the UI.

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

- `src/data/mockData.js` contains isolated analysis, features, transcript, flaw, score, dataset, and evaluation data.
- `src/services/mockServices.js` is the replaceable service boundary for future APIs.
- `src/main.jsx` contains the route shell and reusable analytical UI components.
- `src/styles.css` contains the dark technical design system and responsive layout.
