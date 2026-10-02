# Contrastive Speech Analytics — Full Stack

This archive combines the React/Vite dashboard and the Python speech-scoring engine behind one API. The **Run Evaluation** screen uploads a participant recording, an optional ideal reference, and a transcript; the backend returns rubric scores and time-stamped findings. Existing overview/dataset pages remain demonstration fixtures and are labelled as such.

## Requirements

- Python 3.11 (backend pins PyTorch/torchaudio 2.8.0; do not upgrade torchaudio to 2.9 for this project)
- Node.js 20+ and npm
- On the first forced-alignment run, the MMS_FA model may download roughly 1.5 GB. Cache it with `HF_HOME` if desired.

## Quick start — Windows

1. Install Python 3.11 and Node.js.
2. Open PowerShell in this extracted folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m pip install -r backend\api-requirements.txt
npm --prefix frontend install
```

3. Start the backend in one terminal:

```powershell
$env:PYTHONPATH = (Resolve-Path backend).Path
python -m uvicorn api:app --app-dir backend --host 127.0.0.1 --port 8000
```

4. Start the frontend in a second terminal:

```powershell
npm --prefix frontend run dev
```

Open the local Vite URL printed by npm (normally `http://localhost:5173`) and select **Run Evaluation**. The frontend proxies `/api` to `http://localhost:8000`.

Or run `start_windows.bat` to install frontend packages and start both services in separate terminal windows. Python packages still need to be installed as above.

## Quick start — macOS/Linux

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
python -m pip install -r backend/api-requirements.txt
npm --prefix frontend install
./start_unix.sh
```

`start_unix.sh` runs the API at port 8000 and Vite at port 5173; stop each with Ctrl+C in its terminal.

## Single-origin production preview

Build the Vite app, then start the API. FastAPI serves `frontend/dist` when it exists, including client-side routes:

```bash
npm --prefix frontend run build
# Windows PowerShell:
$env:PYTHONPATH = (Resolve-Path backend).Path
python -m uvicorn api:app --app-dir backend --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000
```

## API

- `GET /api/health` — liveness check
- `GET /api/capabilities` — enabled evaluation modes and upload limits
- `POST /api/evaluate` — `multipart/form-data` with required `participant_audio`, required `transcript`, and optional `reference_audio`
- `GET /docs` — interactive OpenAPI documentation

Each audio upload is limited to 50 MB by default. Set `SPEECH_EVAL_MAX_UPLOAD_BYTES`, `SPEECH_EVAL_TEMP_DIR`, `SPEECH_EVAL_ALIGNMENT_BACKEND` or `SPEECH_EVAL_ALLOW_ALIGNMENT_FALLBACK` before launching the API to adjust behavior. Alignment fallback is on by default and is explicitly noted in the returned result if proportional word timing was used.

If running Vite and the API on different hosts, set `VITE_API_URL` in `frontend/.env.local` to the API origin (for example `http://localhost:8000`). For the bundled same-origin setup leave it empty. Demo mode is off by default; `VITE_DEMO_MODE=true` explicitly selects the existing demo hook fixture.

## Backend caveats

- With a reference audio file, rubric scores and flaw detection are contrastive against that recording; without one, absolute-range heuristics are used.
- The backend README contains its own architecture, benchmark, dataset, and known limitations. Some flaw categories in the current detector are experimental or not yet implemented.
- The source archive contained a Windows virtual environment. It is intentionally **not** included; create a fresh environment for your OS using the instructions above.
- Uploaded temporary files are removed after each request. Evaluations run one at a time per API process to avoid oversubscribing CPU/memory-heavy audio processing.

## Frontend checks

```bash
npm --prefix frontend run build
```
