# 🚀 Deployment Guide — AI Loan Eligibility Prediction System

## Part 1: Upload to GitHub

### Step 1 — Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Fill in:
   - **Repository name:** `Loan-Prediction-ML-App`
   - **Description:** `AI Loan Eligibility Prediction System — ML + Streamlit`
   - **Visibility:** Public
   - **DO NOT** check "Add a README" (you already have one)
3. Click **Create repository**

### Step 2 — Initialize Git and Push

Open your terminal in the project folder and run:

```bash
cd "D:\Coursera\Data Science Interhsip\Loan_Prediction_Project"

git init

git add .

git commit -m "Initial commit: AI Loan Prediction System"

git branch -M main

git remote add origin https://github.com/YOUR_USERNAME/Loan-Prediction-ML-App.git

git push -u origin main
```

> Replace `YOUR_USERNAME` with your actual GitHub username.

### Step 3 — Verify Upload

Go to your GitHub repo URL. You should see:

```
✅ app.py
✅ train.py
✅ eda.py
✅ pdf_generator.py
✅ requirements.txt
✅ README.md
✅ .gitignore
✅ .streamlit/config.toml
✅ data/loan_dataset_2025.csv
✅ models/loan_project_artifacts.pkl
✅ outputs/  (all charts + reports)
```

> **Important:** Make sure `models/loan_project_artifacts.pkl` is uploaded.
> If the file is too large (>100 MB), use [Git LFS](https://git-lfs.github.com/).

---

## Part 2: Deploy on Streamlit Cloud

### Step 1 — Go to Streamlit Cloud

1. Open [share.streamlit.io](https://share.streamlit.io)
2. Click **Sign in with GitHub**
3. Authorize Streamlit to access your GitHub account

### Step 2 — Create a New App

1. Click **"New app"** (top right)
2. Fill in the deployment form:

| Field | Value |
|-------|-------|
| **Repository** | `YOUR_USERNAME/Loan-Prediction-ML-App` |
| **Branch** | `main` |
| **Main file path** | `app.py` |

3. Click **"Deploy!"**

### Step 3 — Wait for Deployment

- Streamlit Cloud will:
  1. Clone your repository
  2. Install packages from `requirements.txt`
  3. Launch `app.py`
- First deployment takes **3–5 minutes**
- You will get a live URL like: `https://your-app-name.streamlit.app`

### Step 4 — Verify the Live App

Check these features work:
- [ ] Dark / Light mode toggle
- [ ] All sidebar inputs render correctly
- [ ] Prediction button returns results
- [ ] Gauge chart and feature importance chart display
- [ ] PDF report downloads successfully
- [ ] AI Chatbot responds to questions

---

## Part 3: Troubleshooting

### "ModuleNotFoundError"
Your `requirements.txt` is missing a package. Add it and push:
```bash
git add requirements.txt
git commit -m "fix: add missing dependency"
git push
```
Streamlit Cloud will auto-redeploy.

### "FileNotFoundError: models/loan_project_artifacts.pkl"
The model file was not pushed to GitHub. Check:
```bash
git lfs track "*.pkl"
git add models/loan_project_artifacts.pkl
git commit -m "fix: add model artifact"
git push
```

### App crashes with no clear error
Check the Streamlit Cloud logs:
1. Go to your app dashboard on [share.streamlit.io](https://share.streamlit.io)
2. Click **"Manage app"** (bottom right of the live app)
3. Click **"Logs"** to see the full error traceback

### App sleeps after inactivity
Free-tier Streamlit Cloud apps go to sleep after a few days of no traffic. Anyone visiting the URL will wake it up automatically (takes ~30 seconds).

---

## Project Structure (Deployment-Ready)

```
Loan_Prediction_Project/
├── .gitignore                       # Excludes venv, __pycache__, OS files
├── .streamlit/
│   └── config.toml                  # Theme + server settings for cloud
├── app.py                           # Main Streamlit application (entry point)
├── pdf_generator.py                 # PDF report engine
├── train.py                         # Model training pipeline
├── eda.py                           # Exploratory Data Analysis
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── data/
│   └── loan_dataset_2025.csv        # Training dataset
├── models/
│   └── loan_project_artifacts.pkl   # Trained model + encoders
└── outputs/
    ├── *.png                        # Training visualizations
    ├── *.csv                        # Model comparison data
    └── *.txt                        # Training reports
```

---

## Quick Reference — Git Commands for Updates

After making changes locally:

```bash
git add .
git commit -m "update: description of change"
git push
```

Streamlit Cloud will **auto-redeploy** within 1–2 minutes after each push.
