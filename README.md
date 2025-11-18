# 🛡️ Fake Review Guard – Chrome Extension + ML API

**Fake Review Guard** is a Chrome extension powered by a locally hosted FastAPI service that detects and flags potentially fake product reviews in real time on sites like **Amazon**, **Yelp**, and **TripAdvisor**.  
It uses a custom trained **scikit-learn** machine learning model, seamlessly integrated with a clean pop-up interface.

---

## ✨ Features

- 🔍 **Real-time Review Scanning:** Detects fake or suspicious reviews as you browse.
- 💬 **On-page Feedback:** Highlights each review with a color-coded badge (fake / genuine).
- 🧠 **Custom ML Model:** Powered by your trained scikit-learn pipeline using multiple vectorizers.
- ⚙️ **FastAPI Backend:** Serves model predictions via a lightweight local REST API.
- 🪄 **Modern UI:** Clean, non-intrusive pop-up for results.
- 🧩 **Modular Design:** Easy to retrain, re-export, or restyle.

---

## 🏗️ Project Architecture
```
fake-review-detector/
│
├── server/ # FastAPI backend (model inference)
│ ├── app.py # Main API server
│ ├── pipeline_4vec.pkl # Combined model + vectorizers
│ ├── requirements.txt
│ └── custom_vec.py # CombinedPrefitVectorizer class
│
└── extension/ # Chrome Extension (Manifest V3)
├── manifest.json
├── background.js
├── content.js
├── options.html
├── options.js
├── styles.css
└── icons/

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/fake-review-guard.git
cd fake-review-guard
```
### 2. Backend Setup (Fast API)
```bash
cd server
python -m venv .venv
.venv\Scripts\activate       # Windows
# or: source .venv/bin/activate  (macOS/Linux)

pip install -r requirements.txt

```
### 3. Start the API Server
```bash
$env:PIPELINE_PATH="pipeline_4vec.pkl"
python -m uvicorn app:app --host 127.0.0.1 --port 8000

```
---
## How It Works
| Component             | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| **Chrome Extension**  | Extracts text of each review from a web page.                   |
| **Background Script** | Sends batches of reviews to the API for prediction.             |
| **FastAPI Server**    | Loads the trained scikit-learn model and returns probabilities. |
| **Content Script**    | Displays polished badges and popups near each review.           |




