<div align="center">

# 🌾 Rahat v2 — AI Farming Assistant

### Smart farming powered by Artificial Intelligence

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-yellow?style=for-the-badge)](https://penguin2004-rahat-v2.hf.space)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple?style=for-the-badge&logo=bootstrap)](https://getbootstrap.com)

</div>

---

## 📖 About Rahat

**Rahat** (राहत) means *relief* in Hindi and Marathi — relief for Indian farmers through the power of AI.

Rahat v2 is a full-stack AI-powered web application designed to assist Indian farmers with three core features:

- 🌱 **Crop Recommendation** — Get the best crop suggestion based on your soil nutrients and real-time weather
- 💊 **Fertilizer Suggestion** — Know exactly what fertilizer your soil needs, powered by Groq AI
- 🔬 **Plant Disease Detection** — Upload a leaf photo and instantly detect plant disease using deep learning

The app supports both **English** and **Devanagari (Hindi/Marathi)** making it accessible to farmers across India.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🌾 Crop Recommendation | ML model trained on soil NPK, pH, temperature, humidity and rainfall data |
| 💊 Fertilizer Advisor | Groq AI (LLaMA 3.3 70B) generates detailed fertilizer + bio fertilizer advice |
| 🔬 Disease Detection | ResNet50 CNN model fine-tuned on PlantVillage dataset — detects 15 conditions |
| 🌦️ Weather Integration | OpenWeatherMap API auto-fills temperature, humidity and rainfall via GPS |
| 🌐 Bilingual UI | English + Hindi/Marathi (Devanagari) throughout the interface |
| 📱 Responsive Design | Works on mobile, tablet and desktop |

---

## 🛠️ Tech Stack

### Backend
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

### AI / ML
![Groq](https://img.shields.io/badge/Groq_AI-LLaMA_3.3_70B-black?style=flat)
![ResNet50](https://img.shields.io/badge/ResNet50-Fine--tuned-orange?style=flat)
![PlantVillage](https://img.shields.io/badge/Dataset-PlantVillage-green?style=flat)

### Frontend
![Bootstrap](https://img.shields.io/badge/Bootstrap_5-7952B3?style=flat&logo=bootstrap&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

### APIs & Deployment
![OpenWeatherMap](https://img.shields.io/badge/OpenWeatherMap-API-orange?style=flat)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Spaces-yellow?style=flat&logo=huggingface)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Google Drive](https://img.shields.io/badge/Google_Drive-Model_Storage-blue?style=flat&logo=googledrive)

---

## 🚀 Live Demo

👉 **[https://penguin2004-rahat-v2.hf.space](https://penguin2004-rahat-v2.hf.space)**

---

## 📸 Screenshots

> Add screenshots of your app here after deployment
> 
> Suggested screenshots:
> - Home page
> - Crop recommendation result
> - Fertilizer AI advice
> - Disease detection result

---

## 🗂️ Project Structure
```
rahat-v2/
├── app.py                          # Main Flask application
├── Dockerfile                      # Docker config for HF Spaces
├── requirements.txt                # Python dependencies
├── Procfile                        # Gunicorn start command
├── app/
│   ├── templates/
│   │   ├── base.html               # Base layout (navbar + footer)
│   │   ├── index.html              # Home page
│   │   ├── crop.html               # Crop recommendation form
│   │   ├── crop_result.html        # Crop result page
│   │   ├── fertilizer.html         # Fertilizer form
│   │   ├── fertilizer_result.html  # Fertilizer AI advice result
│   │   ├── disease.html            # Disease detection upload
│   │   └── disease_result.html     # Disease result page
│   └── static/
│       ├── css/style.css           # Custom styles
│       └── uploads/                # Uploaded leaf images (temp)
└── models/                         # ML models (auto-downloaded from Google Drive)
    ├── crop_model.pkl
    ├── crop_scaler.pkl
    ├── crop_label_encoder.pkl
    ├── fertilizer_model.pkl
    ├── fertilizer_scaler.pkl
    ├── fertilizer_le_*.pkl
    ├── disease_model_final.keras
    ├── disease_class_labels.pkl
    └── disease_class_info.pkl
```

---

## ⚙️ Local Installation

### Prerequisites
- Python 3.11
- Git

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Shwetatate/RAHAT_V2.git
cd RAHAT_V2
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root folder:
```
OPENWEATHER_API_KEY=your_openweathermap_api_key
GROQ_API_KEY=your_groq_api_key
RAHAT_SECRET_KEY=any_random_secret_string
```

**5. Run the app**
```bash
python app.py
```

Open `http://localhost:5000` in your browser.

> **Note:** ML models will auto-download from Google Drive on first run. This may take a few minutes depending on your internet speed.

---

## 🔑 API Keys Setup

You need 2 free API keys to run this app:

### 1. OpenWeatherMap API (Free)
1. Go to [openweathermap.org](https://openweathermap.org/api)
2. Sign up for a free account
3. Go to **API Keys** tab
4. Copy your default API key
5. Add to `.env` as `OPENWEATHER_API_KEY`

### 2. Groq AI API (Free)
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Go to **API Keys** → **Create API Key**
4. Copy the key
5. Add to `.env` as `GROQ_API_KEY`

---

## 🤖 ML Models

| Model | Type | Purpose | Accuracy |
|---|---|---|---|
| Crop Model | Random Forest | Recommend best crop from soil + weather | ~99% |
| Fertilizer Model | Random Forest | Suggest fertilizer type | ~98% |
| Disease Model | ResNet50 (Fine-tuned) | Detect 15 plant diseases | ~98.4% |

### Supported Crops
🌾 Rice • 🌽 Maize • 🫘 Chickpea • 🫘 Kidney Beans • 🫘 Lentil • 🥭 Mango • 🍌 Banana • 🍇 Grapes • 🍉 Watermelon • 🥥 Coconut • ☕ Coffee and more

### Supported Diseases (15 conditions)
- **Pepper** — Bacterial Spot, Healthy
- **Potato** — Early Blight, Late Blight, Healthy
- **Tomato** — Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

---

## 🌐 Deployment

This app is deployed on **Hugging Face Spaces** using Docker.

### Deploy your own copy
1. Fork this repository
2. Create a new Space on [huggingface.co](https://huggingface.co)
3. Select **Docker** as SDK
4. Push your code to the Space
5. Add environment variables in Space Settings → Repository Secrets

---

## 👨‍💻 Author

**Shweta Tate**

[![GitHub](https://img.shields.io/badge/GitHub-Shwetatate-black?style=flat&logo=github)](https://github.com/Shwetatate)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

🌱 Built with ❤️ for farmers of India

**Rahat — राहत — Relief**

</div>