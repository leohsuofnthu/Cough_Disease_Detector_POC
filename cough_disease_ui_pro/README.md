# Cough Disease Detector (UI Pro)

Professional, cross-platform Expo (TypeScript) app to record cough audio and run inference via an API, with a polished light medical theme and results visualization.

## Tech
- Expo SDK 51, TypeScript
- Libraries: expo-av, expo-linear-gradient, react-native-paper, react-native-svg, recharts (web-only)

## Features
- Record / Stop & Analyze flow with timer and pulse animation
- Multipart upload to inference API (defaults to `EXPO_PUBLIC_API_URL`, field: `file`)
- Mock prediction fallback when API is unreachable
- Result card with top prediction, confidence, timestamp, duration
- Horizontal probability bar chart (native via `react-native-svg`, web via `recharts`) showing 7 ICBHI classes (Normal, COPD, Heart Disease, Bronchiectasis, Pneumonia, URTI, LRTI)

## Setup
1. Install dependencies:
   - `npm install`
2. Start the app:
   - `npx expo start`
   - Press `i` for iOS simulator, `a` for Android, or `w` for web.

## Notes
- Mobile uses `expo-av` for recording; Web uses `MediaRecorder`.
- Replace the endpoint in `utils/api.ts` with your real API. Response should expose the same ICBHI class keys as the backend. When using Docker Compose, the app reads `EXPO_PUBLIC_API_URL` (defaults to `http://backend:8000/v1/infer`).
- Colors:
  - Background: `#F7FAFC`
  - Primary: `#2563EB`
  - Accent: `#10B981`

## Docker (web dev)
- `docker compose up --build` (from repo root) launches Expo web at `http://localhost:3000` alongside the backend.

## Structure
```
cough_disease_ui_pro/
├── App.tsx
├── package.json
├── tsconfig.json
├── babel.config.js
├── app.json
├── components/
│   ├── RecordButton.tsx
│   ├── ResultCard.tsx
│   ├── ProbabilityChart.tsx
│   └── LoaderOverlay.tsx
├── utils/
│   ├── api.ts
│   └── time.ts
├── assets/
│   └── icons/ (placeholder icons)
└── README.md
```
