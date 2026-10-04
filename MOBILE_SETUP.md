# Body Lab mobile app

The existing `bodylab/` intelligence and Streamlit app stay intact. The mobile client talks to a small FastAPI layer.

## 1. Python API

From the project root, with `.venv` activated:

```bash
python -m pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Check it in a browser at `http://127.0.0.1:8000/api/health`.

## 2. Find your Mac's Wi-Fi address

For a physical iPhone/Android phone, the phone cannot use `127.0.0.1` to reach your Mac. Keep the phone and Mac on the same Wi-Fi and run:

```bash
ipconfig getifaddr en0
```

If it prints, for example, `192.168.1.25`, create `mobile/.env` with:

```text
EXPO_PUBLIC_API_URL=http://192.168.1.25:8000
```

For iOS Simulator on the same Mac, `EXPO_PUBLIC_API_URL=http://127.0.0.1:8000` is fine.

## 3. Expo mobile app

Install Node.js if needed, then:

```bash
cd mobile
cp .env.example .env
# Edit .env if using a physical phone.
npm install
npx expo start
```

Install **Expo Go** on the phone and scan the QR code shown by Expo. Keep the FastAPI terminal running while using the app.

## 4. Demo data

If no participant exists yet, from the project root run:

```bash
python scripts/prepare.py --synthetic
```

The app automatically selects the first available participant (normally `S01`).

## API routes

- `GET /api/health`
- `GET /api/participants`
- `GET /api/today/{pid}`
- `GET /api/cases/{pid}`
- `GET /api/discoveries/{pid}`
- `GET /api/hypotheses/{pid}`
- `POST /api/chat`

The Gemini key stays only in the Python backend `.env`; it is never placed in the mobile app.
