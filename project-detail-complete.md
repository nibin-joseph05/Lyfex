## Lyfex – Real‑Time, Contactless Health Monitoring (Project Report)

### Objective / Problem Statement
Traditional health checks often require dedicated devices and visits, which make frequent self‑monitoring difficult. Lyfex aims to provide a quick, contactless wellness snapshot using only a phone’s camera. The goal is to estimate key physiological and psychological signals (e.g., heart rate, breathing rate, stress, emotion, alertness) in real‑time and present them in a simple, user‑friendly interface while keeping data within the user’s local environment.

### Tools and Libraries Used
- Mobile App (Frontend)
  - React Native + Expo: cross‑platform mobile UI
  - Expo Router: navigation and routing
  - React Native Reanimated: lightweight UI animations
  - Expo Camera: camera access for frame capture
  - TypeScript: type‑safe app code

- Backend (Server)
  - FastAPI (Python): high‑performance API and WebSocket server
  - OpenCV: computer vision pipeline for face detection and signal extraction
  - NumPy: numerical processing and signal handling
  - Uvicorn: ASGI server for FastAPI

- Communication
  - WebSocket: low‑latency streaming of camera frames and real‑time results

- Development & Build
  - Node.js and npm (frontend)
  - Python 3.10+ and pip (backend)

### Detailed Explanation of the Workflow
1) User opens the mobile app and taps “Start scan”. The app requests camera permission and begins capturing low‑rate frames (optimized for battery and network).

2) The app sends frames to the local backend over a WebSocket connection. Each message contains a base64 image plus metadata (timestamp, estimated FPS). The phone and server must be on the same Wi‑Fi network.

3) The backend (FastAPI + OpenCV) processes the frame:
   - Face detection and landmarks (to ensure a valid face view).
   - Signal extraction from facial regions for physiological metrics:
     - Heart Rate estimation (photoplethysmography/PPG‑like approach using color intensity changes in a facial ROI such as forehead/cheeks).
     - Respiratory Rate estimation (motion/optical flow cues and/or color variation near nostrils and upper chest area when in frame).
   - Psychological/behavioral estimation (lightweight proxies):
     - Emotion classification (categorical estimate, e.g., Neutral/Happy/etc.).
     - Stress level proxy derived from combined signals (e.g., heart rate variability proxy, facial tension heuristics, respiration irregularity).
     - Alertness/Fatigue proxies using eyelid behavior and eye/face stability signals where available.
   - Additional indicators (when data supports it): hydration/skin approximations, facial asymmetry checks, tremor presence, cognitive load proxy, pain proxy.

4) The backend returns a JSON payload over WebSocket including:
   - realtime_metrics: key numbers (heart_rate, respiratory_rate, stress_level, emotion, confidence, face_detected)
   - face_detection: bounding box, landmarks (if available), and quality status
   - analysis_data: extended fields (e.g., fatigue_level, hrv proxy, cognitive_load, hydration/skin assessments)
   - timestamp: ISO time of the reading

5) The app updates the UI in real time:
   - Displays heart rate, breathing rate, emotion, stress level, and confidence.
   - Shows a simple face guide (bounding box/quality) to help the user adjust lighting and distance.
   - Can show friendly messages/alerts when values are out of normal ranges.

6) Session lifecycle:
   - Start/Stop controls the streaming. When stopped, the app can summarize the last session (duration, number of readings) and return to the home screen.
   - By default, the template focuses on real‑time display; persistent history can be added later.

#### Data Flow (High Level)
- Camera (phone) → WebSocket (phone ↔ local server) → OpenCV/Models (server) → JSON results → UI rendering (phone)

#### Privacy Considerations
- The server typically runs on the user’s own computer in the same local network as the phone; data is not sent to third‑party services by default.
- Users can stop the server at any time to stop all processing.

### Screenshots of Key Outputs (Placeholders)
Add your images in this section. Suggested items to capture:
- Home screen / Start Scan button
- Live detection screen with heart rate, breathing rate, emotion, stress
- Face guide overlay (bounding box & quality)
- An example of the real‑time log/values changing

Example placeholders (replace with your own images):
- ![Home Screen](screenshots/home.png)
- ![Detection Live Metrics](screenshots/detection.png)
- ![Face Box & Quality](screenshots/face_box.png)

### Conclusion
Lyfex demonstrates that a phone camera, combined with a small local server, can provide quick, contactless wellness estimates. While this is not a medical device, it can help users get immediate feedback on heart rate, breathing, stress, and alertness, encouraging better self‑awareness. The architecture keeps data within the user’s environment by default, and the system can be extended with trends, alerts, and calibration.

### Learning Outcomes
- Built a cross‑platform mobile interface with React Native + Expo.
- Implemented a real‑time camera → server → results feedback loop using WebSockets.
- Applied computer vision (OpenCV) and simple signal processing to estimate physiological metrics from the face.
- Designed a clean, user‑friendly UI for live metrics with guidance and status.
- Understood constraints that affect accuracy (lighting, motion, camera quality) and ways to improve UX.
- Identified clear next steps: history storage, richer alerting, calibration, and potential on‑device inference for modern phones.


