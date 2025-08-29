## Lyfex – Camera-based Health Snapshot (Simple Guide)

Lyfex turns your phone into a contactless wellness monitor. Using the front camera, it estimates key signals like heart rate and breathing, then shows an easy health snapshot with tips. This guide is written for everyone (no tech background needed).

### What Lyfex does
- Live health snapshot from your face using the phone camera
- Shows: heart rate, breathing rate, stress, emotion, alertness/fatigue, hydration/skin (approx.)
- Simple indicators and messages when something needs attention
- Runs on your own device and local network by default (no accounts)

### How it works (in plain words)
1. The mobile app opens your camera and sends light snapshots to your own local server.
2. The local server analyzes your face with computer vision (OpenCV) and AI models.
3. The server instantly sends results back (heart rate, breathing, stress, etc.).
4. The app displays your live numbers and helpful messages.

Nothing is sent to outside services unless you change the settings. Your phone and computer must be on the same Wi‑Fi.

### What you need
- A phone with a front camera (Android recommended for this project template)
- A computer on the same Wi‑Fi to run the local server
- Good lighting and a steady view of your face

## Quick start

### 1) Start the backend (local server)
Requirements: Python 3.10+.

Steps:
1. Open a terminal in `backend/`.
2. (Windows) Create and activate a virtual environment (optional if `venv` already exists):
   - `python -m venv venv`
   - `venv\Scripts\activate`
3. Install packages:
   - `pip install -r requirements.txt`
4. Start the server:
   - `uvicorn main:app --host 0.0.0.0 --port 8000`
5. Keep this running. Visit `/health` to see the status message.

### 2) Start the mobile app (frontend)
Requirements: Node.js and Expo.

Steps:
1. In another terminal, go to `frontend/`.
2. Install packages: `npm install`.
3. Tell the app where your backend is (same Wi‑Fi):
   - On Windows (Command Prompt):
     - `set EXPO_PUBLIC_BACKEND_URL=http://YOUR_PC_LOCAL_IP:8000`
   - Find your PC IP with `ipconfig` (look for IPv4), for example `192.168.1.23`.
4. Start the app: `npm run start` and follow Expo’s instructions to open it on your phone or emulator.

Tips:
- If you see “WebSocket not ready”, double‑check the backend is running and the URL is correct.
- Improve lighting and keep your face centered for better results.

## Using Lyfex
- Sit in good lighting, keep your face within the frame.
- Tap “Start scan”. Watch the live numbers and status messages.
- If quality is low, adjust lighting and distance from the camera.

## Privacy
- Analysis runs on your own local server by default, and results are sent back to your app.
- No accounts. No external cloud required.
- Stop the server anytime to stop all processing.

## Common questions
- Does this diagnose medical conditions?
  - No. It’s for wellness insights only and not a medical device.
- How accurate is it?
  - It provides helpful estimates. Lighting, camera quality, and movement affect results.
- Can it work offline?
  - The mobile app needs your local server running. Without it, live analysis won’t run.
- Is any data stored?
  - The current template focuses on live viewing. Saving history can be added later.
- Does it send data to the internet?
  - Not unless you change settings. By default, it stays on your local network.

## What’s inside (for the curious)
- Frontend: React Native + Expo (in `frontend/`)
  - Screens: Home and Real‑time Detection
  - Connects to the backend via WebSocket
- Backend: FastAPI + OpenCV (in `backend/`)
  - Receives camera frames, extracts signals, returns metrics
  - Components for heart rate, breathing, stress, fatigue, skin, etc.

## Troubleshooting
- “WebSocket not ready” → Start the backend and set `EXPO_PUBLIC_BACKEND_URL` correctly.
- “No face detected” → Improve lighting, center your face, remove obstructions.
- Jittery readings → Hold still a few seconds and ensure a steady network connection.

## Future ideas
- Save sessions and trends (history)
- Alert badges and notifications
- Calibration/baseline setup
- Fully on‑device analysis on modern phones

If you’re not technical: follow “Quick start” step by step. If something doesn’t work, check “Troubleshooting” or ask for help with your device model and network details.


