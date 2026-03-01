<!-- Created by Codex - Section 2 -->

# Medical LLM Assistant Frontend

Angular 17 single-page application for the FastAPI-backed medical literature assistant.

## Commands

```bash
npm ci
npm start
npm run build -- --configuration production
npm test -- --watch=false --browsers=ChromeHeadless
```

## Notes

- Development requests to `/api` are proxied to `http://localhost:8000` via `src/proxy.conf.json`.
- Theme state and runtime controls are persisted in `localStorage`.
- Production builds are emitted under `dist/medical-llm-assistant/browser`.
