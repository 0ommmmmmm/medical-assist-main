# Medical Assist — Android App
### Complete Setup & Build Guide

> ⚠️ **DISCLAIMER**: For educational purposes ONLY. Not a substitute for professional medical advice.

---

## Architecture Overview

```
Medical-Assist-Main/
├── src/                          ← Original Python ML backend
│   ├── api/main.py               ← FastAPI (original)
│   └── models/predictors.py      ← ML inference classes
├── backend_enhanced/
│   └── main.py                   ← Enhanced API (new: /chat, /auth, mobile CORS)
└── flutter_app/                  ← Flutter Android App
    ├── lib/
    │   ├── main.dart             ← App entry point
    │   ├── theme/app_theme.dart  ← Colors, typography, component styles
    │   ├── models/models.dart    ← Data models (DiabetesInput, PredictionResult...)
    │   ├── services/
    │   │   ├── api_service.dart  ← HTTP calls to FastAPI backend
    │   │   ├── auth_service.dart ← Firebase Auth (login/signup)
    │   │   ├── offline_service.dart ← Hive caching + connectivity
    │   │   └── voice_service.dart   ← Speech-to-Text + TTS
    │   ├── screens/
    │   │   ├── auth_screen.dart      ← Login / Signup
    │   │   ├── home_screen.dart      ← Dashboard
    │   │   ├── diabetes_screen.dart  ← 3-step form + sliders
    │   │   ├── pneumonia_screen.dart ← X-ray image upload
    │   │   ├── chat_screen.dart      ← AI chat + voice input
    │   │   ├── result_screen.dart    ← Analysis results
    │   │   └── history_screen.dart   ← Past results
    │   ├── widgets/app_widgets.dart  ← Reusable UI components
    │   └── utils/router.dart         ← go_router navigation
    └── android/                      ← Android platform config
```

---

## Why Flutter?

| Feature | Flutter | React Native | Native Kotlin |
|---|---|---|---|
| Single codebase (Android + iOS) | ✅ | ✅ | ❌ |
| Performance (no JS bridge) | ✅ Fast | ⚠️ Slower | ✅ Fastest |
| Camera / Voice / Permissions | ✅ | ✅ | ✅ |
| Medical UI polish | ✅ Easy | ✅ Moderate | ✅ Full control |
| Beginner-friendly | ✅ | ⚠️ | ❌ |
| Firebase integration | ✅ | ✅ | ✅ |
| **Recommendation** | **✅ Best for this project** | | |

---

## Part 1: Backend Setup

### Step 1 — Install backend dependencies

```bash
cd Medical-Assist-Main
pip install -r backend_enhanced/requirements_enhanced.txt
```

### Step 2 — Train the models (if not already trained)

```bash
python src/training/train_diabetes.py
python src/training/train_pneumonia.py
```

### Step 3 — (Optional) Add Claude AI for the chat endpoint

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

Without this, the `/chat` endpoint uses a built-in rule-based fallback.

### Step 4 — Run the enhanced backend

```bash
# From the project root:
uvicorn backend_enhanced.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ Visit `http://localhost:8000/docs` to see the Swagger UI.

---

## Part 2: Flutter App Setup

### Prerequisites

1. **Install Flutter SDK** (3.19+):
   ```bash
   # macOS / Linux
   git clone https://github.com/flutter/flutter.git
   export PATH="$PATH:`pwd`/flutter/bin"
   flutter doctor
   
   # Windows: download from flutter.dev
   ```

2. **Install Android Studio** with:
   - Android SDK (API 34)
   - Android Emulator or connect a real device via USB

3. **Verify setup**:
   ```bash
   flutter doctor -v
   ```
   All checkmarks should be green (except iOS if on Windows/Linux).

### Step 1 — Set up the Flutter project

```bash
cd Medical-Assist-Main/flutter_app

# Install all dependencies
flutter pub get

# Generate Hive type adapters and code
flutter pub run build_runner build --delete-conflicting-outputs
```

### Step 2 — Configure Firebase (for Auth)

**Option A: Use Firebase (recommended for production)**

1. Go to [console.firebase.google.com](https://console.firebase.google.com)
2. Create a new project: "Medical Assist"
3. Add an Android app with package name: `com.example.medical_assist`
4. Download `google-services.json`
5. Place it at: `flutter_app/android/app/google-services.json`
6. Uncomment Firebase lines in `android/app/build.gradle`
7. Uncomment Firebase imports in `lib/main.dart`
8. Enable Email/Password auth in Firebase Console → Authentication → Sign-in method

**Option B: Skip Firebase (use backend auth only)**

The backend's `/auth/register` and `/auth/login` endpoints provide
JWT-based auth without Firebase. To use this instead:

```dart
// In lib/services/auth_service.dart, replace Firebase calls with:
final res = await _dio.post('/auth/login', data: {'email': email, 'password': password});
final token = res.data['token'];
await _offlineService.saveAuthToken(token);
```

### Step 3 — Configure the API URL

In `lib/services/api_service.dart`, set your backend URL:

```dart
class ApiConfig {
  // For Android Emulator connecting to your machine:
  static const String baseUrl = 'http://10.0.2.2:8000';
  
  // For a real device on same WiFi network:
  // Find your machine IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
  // static const String baseUrl = 'http://192.168.1.X:8000';
  
  // For a deployed server:
  // static const String baseUrl = 'https://your-api.railway.app';
}
```

---

## Part 3: Running the App

### Run on Android Emulator

```bash
# Start an emulator from Android Studio, then:
flutter run
```

### Run on a Real Android Device

1. Enable **Developer Options** on your device:
   - Go to Settings → About Phone
   - Tap "Build Number" 7 times
   
2. Enable **USB Debugging** in Developer Options

3. Connect via USB and accept the debug prompt

4. Run:
   ```bash
   flutter devices          # List connected devices
   flutter run -d <device_id>
   ```

---

## Part 4: Building the APK

### Debug APK (for testing, larger file)

```bash
flutter build apk --debug
# Output: build/app/outputs/flutter-apk/app-debug.apk
```

### Release APK (for distribution, optimized)

```bash
flutter build apk --release --split-per-abi
# Outputs separate APKs for arm64, arm32, x86_64:
# build/app/outputs/flutter-apk/app-arm64-v8a-release.apk  ← Use this for most modern phones
# build/app/outputs/flutter-apk/app-armeabi-v7a-release.apk
```

### App Bundle (for Google Play Store)

```bash
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

### Install APK on a connected device

```bash
flutter install
# Or manually:
adb install build/app/outputs/flutter-apk/app-debug.apk
```

---

## Part 5: Deploying the Backend (Optional)

### Option A: Railway (easiest, free tier available)

```bash
# Install Railway CLI
npm install -g @railway/cli

# From project root:
railway login
railway init
railway up
```

Set environment variables in Railway dashboard:
- `ANTHROPIC_API_KEY` = your key

### Option B: Render.com

1. Push your code to GitHub
2. Create a new Web Service on render.com
3. Set build command: `pip install -r backend_enhanced/requirements_enhanced.txt`
4. Set start command: `uvicorn backend_enhanced.main:app --host 0.0.0.0 --port $PORT`

### Option C: Local Network (development)

Run the backend on your machine and connect via LAN:
```bash
uvicorn backend_enhanced.main:app --host 0.0.0.0 --port 8000
```
Use your machine's local IP in the Flutter app config.

---

## Features Summary

| Feature | Status | Notes |
|---|---|---|
| Login / Signup | ✅ | Firebase Auth or backend JWT |
| Home Dashboard | ✅ | Recent results, quick actions |
| Diabetes Assessment | ✅ | 3-step form with sliders + reference guides |
| Pneumonia X-Ray | ✅ | Camera or gallery upload |
| AI Chat | ✅ | Claude API or rule-based fallback |
| Voice Input | ✅ | Speech-to-text via device mic |
| Text-to-Speech | ✅ | Reads AI responses aloud |
| Offline Mode | ✅ | Hive cache, offline chat fallback |
| Result History | ✅ | Filterable, persistent |
| Modern UI | ✅ | Gradient cards, animations |
| Android APK | ✅ | minSdk 23, targets 34 |

---

## Troubleshooting

### "Cannot connect to backend"
- Make sure the FastAPI server is running on port 8000
- If using emulator, use `10.0.2.2` not `localhost`
- If using real device, use your machine's LAN IP (e.g., `192.168.1.5`)
- Check that `usesCleartextTraffic="true"` is in AndroidManifest.xml for HTTP

### "Model not loaded" error
- Run the training scripts first:
  ```bash
  python src/training/train_diabetes.py
  python src/training/train_pneumonia.py
  ```

### "Firebase not initialized"
- Make sure `google-services.json` is in `android/app/`
- Or comment out Firebase code and use backend auth instead

### Build errors
```bash
flutter clean
flutter pub get
flutter pub run build_runner build --delete-conflicting-outputs
flutter run
```

### Voice input not working
- Make sure microphone permission is granted in Android Settings → Apps → Medical Assist → Permissions
- Voice input requires an internet connection on most devices

---

## Project Dependencies

```yaml
# Core
flutter_riverpod: ^2.5.1   # State management
go_router: ^13.2.0          # Navigation
dio: ^5.4.3                 # HTTP client

# Firebase
firebase_auth: ^4.19.6
cloud_firestore: ^4.17.4

# Local Storage
hive_flutter: ^1.1.0        # Fast offline storage
shared_preferences: ^2.2.3  # Simple key-value store

# Media
image_picker: ^1.1.2        # Camera & gallery
speech_to_text: ^6.6.2      # Voice input
flutter_tts: ^4.0.2         # Text-to-speech

# UI
flutter_animate: ^4.5.0     # Smooth animations
google_fonts: ^6.1.0        # Inter font
fl_chart: ^0.68.0           # Charts (for future use)
```

---

## Security Notes

- **Never hardcode API keys** in the Flutter app
- Use environment variables for the backend API key
- For production, enable HTTPS on your backend
- The backend's in-memory auth is for demo only — use Firebase or a real DB in production
- All user data stored locally via Hive is on-device only

---

*Built with Flutter 3.x | FastAPI 0.110+ | Python 3.11+*
