<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Plotly-5.0+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

<h1 align="center">🏦 AI Loan Eligibility Prediction System</h1>

<p align="center">
  <strong>An end-to-end machine learning web application that predicts loan approval status in real-time.</strong>
</p>

<p align="center">
  This production-grade system features a multi-model training pipeline, a professional Streamlit dashboard with an AI chatbot, and automated PDF report generation.
</p>

---

## 📋 Table of Contents

- [🚀 About the Project](#-about-the-project)
- [🎥 Live Demo](#-live-demo)
- [🔗 Project Links](#-project-links)
- [✨ Key Features](#-key-features)
- [🛠️ Tech Stack](#-tech-stack)
- [⚙️ How It Works](#️-how-it-works)
- [🧠 Model Training Details](#-model-training-details)
- [📊 Outputs Generated](#-outputs-generated)
- [️ UI Overview](#️-ui-overview)
- [🚀 Getting Started](#️-getting-started)
- [🚢 Deployment](#-deployment)
- [🔮 Future Improvements](#-future-improvements)
- [👨‍💻 Author](#-author)
- [📜 License](#-license)

---

## 🚀 About the Project

The **AI Loan Eligibility Prediction System** is a complete data science project that covers the entire ML lifecycle — from exploratory data analysis to model training, evaluation, and deployment as an interactive web application.

The system trains and compares multiple classification models, automatically selects the best performer based on ROC AUC, and serves predictions through a professional **Streamlit dashboard**. The dashboard includes real-time analytics, an AI chatbot assistant for explaining results, and downloadable multi-page PDF assessment reports.

This project was developed to demonstrate proficiency in end-to-end machine learning workflows, from data analysis and model development to full-stack deployment and MLOps principles.

---

## 🎥 Live Demo

A live version of this application is deployed on Streamlit Cloud. You can interact with the full system without any local setup.

**[➡️ Access the Live Demo Here](https://loan-approval-prediction-system-ai.streamlit.app/)**

In the live demo, you can:
-   **Submit a Loan Application**: Fill out the form with sample data to generate a real-time prediction.
-   **Interact with the AI Chatbot**: Ask questions about the prediction, risk factors, or how to improve an applicant's profile.
-   **Generate a PDF Report**: Download a professional, multi-page PDF summary of the loan assessment.
-   **Explore the Dashboard**: View the model comparison dashboard and interactive charts.
-   **Switch Themes**: Toggle between dark and light mode to see the custom UI in action.

---

## 🔗 Project Links

- **GitHub Repository:** https://github.com/your-username/Loan-Prediction-ML-App
- **Live Application:** [➡️ Access the Live Demo Here](https://loan-approval-prediction-system-ai.streamlit.app/)

---

## ✨ Key Features

| Feature | Description |
|:--------------------------------|:------------------------------------------------------------------------------------------------|
| 🤖 **Real-Time Prediction** | Predicts loan approval status instantly using the best-performing trained model. |
| 🏆 **Automated Model Selection** | Trains and evaluates multiple models, automatically selecting the best one for deployment. |
| 📊 **Interactive Dashboard** | A comprehensive Streamlit dashboard with KPI cards, charts, and model performance metrics. |
| 📈 **Probability & Risk Scoring** | Displays approval probability with an interactive gauge chart and classifies risk (Low, Medium, High). |
| 🧠 **AI Decision Insights** | Provides feature importance charts to explain which factors most influenced the prediction. |
| 💬 **AI Chatbot Assistant** | A built-in conversational assistant to explain loan status, risk factors, and improvement strategies. |
| 📄 **PDF Report Generation** | Generates professional, multi-page PDF reports with applicant data, AI insights, and visual analytics. |
| 🎨 **Premium UI/UX** | Custom-themed interface with dark/light mode toggle and a clean, professional layout. |
| ⚙️ **End-to-End Pipeline** | Includes scripts for data analysis (`eda.py`), model training (`train.py`), and deployment (`app.py`). |

---

## 🛠️ Tech Stack

| Category | Technologies |
|:----------------|:--------------------------------------------------------------------------------|
| **Core Language** | Python 3.10+ |
| **Data Science** | Pandas, NumPy, Scikit-Learn |
| **Web Framework** | Streamlit |
| **Data Viz** | Plotly (Interactive), Matplotlib, Seaborn (Static) |
| **PDF Generation**| ReportLab |
| **MLOps** | Pickle (Serialization), Git, GitHub |

---

## ⚙️ How It Works

The project follows a structured machine learning pipeline:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Loading    │────▶│  Data Cleaning   │────▶│  Encoding       │
│  (CSV Dataset)   │     │  (Missing Values)│     │  (LabelEncoder) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Deployment     │◀────│  Best Model      │◀────│  Feature        │
│  (Streamlit)    │     │  Selection       │     │  Engineering    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              ▲                         │
                              │                         ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Model           │◀────│  Model Training │
                        │  Evaluation      │     │  (3 Algorithms) │
                        └─────────────────┘     └─────────────────┘
```

### Pipeline Steps

1. **Data Loading** — Load the loan dataset (`loan_dataset_2025.csv`) with applicant financial and personal attributes
2. **Data Cleaning** — Handle missing values using median (numeric) and mode (categorical) imputation
3. **Encoding** — Transform categorical features using `LabelEncoder` for model compatibility
4. **Feature Engineering** — Apply `log1p` transformations on `monthly_income` and `loan_amount` to reduce skewness
5. **Model Training** — Train three classifiers with optimized hyperparameters
6. **Model Evaluation** — Evaluate using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation
7. **Best Model Selection** — Automatically select the model with the highest accuracy score
8. **Deployment** — Serve predictions via an interactive Streamlit web application with real-time analytics

---

## 🧠 Model Training Details

Three classification models are trained and compared:

| Model | Configuration | Details |
|-------|--------------|---------|
| **Logistic Regression** | Default parameters | Linear baseline classifier for binary loan prediction |
| **Decision Tree** | `max_depth=8` | Non-linear classifier with controlled depth to prevent overfitting |
| **Random Forest** | `n_estimators=300` | Ensemble of 300 decision trees for robust predictions |

### Training Configuration

- **Train/Test Split:** 80/20 stratified split (preserves class distribution)
- **Encoding:** `LabelEncoder` for all categorical features
- **Feature Transforms:** `log1p` applied to `monthly_income` and `loan_amount`
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-Validation:** K-fold cross-validation for model stability assessment
- **Artifact Storage:** Best model, feature list, and encoders saved to `models/loan_project_artifacts.pkl`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/Loan_Prediction_Project.git
cd Loan_Prediction_Project
```

**2. Create a virtual environment**

```bash
python -m venv venv
```

**3. Activate the virtual environment**

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run Exploratory Data Analysis** *(optional)*

```bash
python eda.py
```

**6. Train the ML models**

```bash
python train.py
```

> This will train all three models, evaluate them, select the best model, and save artifacts to the `models/` and `outputs/` directories.

**7. Launch the Streamlit app**

```bash
streamlit run app.py
```

> The app will open in your browser at `http://localhost:8501`

---

## 📊 Outputs Generated

After running the training pipeline (`train.py`), the following outputs are generated:

### 📈 Visualizations

| Output | Description |
|--------|-------------|
| `confusion_matrix.png` | Confusion matrix heatmap for the best model |
| `roc_curve.png` | ROC curve with AUC score |
| `feature_importance.png` | Top feature importance bar chart |
| `accuracy_comparison.png` | Accuracy comparison across all models |
| `precision_comparison.png` | Precision comparison across all models |
| `recall_comparison.png` | Recall comparison across all models |
| `f1_comparison.png` | F1-score comparison across all models |
| `model_dashboard.png` | Combined 4-in-1 metrics dashboard |
| `final_model_ranking.png` | Final ranking of all trained models |
| `precision_recall_curve.png` | Precision-recall tradeoff curve |

### 📋 Reports & Data

| Output | Description |
|--------|-------------|
| `model_comparison.csv` | Side-by-side model metrics comparison |
| `model_leaderboard.csv` | Ranked model leaderboard |
| `classification_report.txt` | Detailed precision, recall, F1 per class |
| `training_summary.txt` | Complete training session summary |
| `final_ai_report.txt` | AI-generated analysis report |
| `loan_report.pdf` | Professional 8-page PDF report with full applicant profile |

---

## 🖥️ UI Overview

The Streamlit web application provides a professional, fintech-grade interface:

### 🎛️ Sidebar — Applicant Input Form

- **Personal Info:** Full name, CNIC, phone number, email
- **Demographics:** Age, gender, marital status, education level, employment status
- **Financial Details:** Annual income, monthly income, credit score, debt-to-income ratio
- **Loan Details:** Loan amount, loan purpose, interest rate, grade/subgrade, loan term (12–60 months)
- **🚀 Predict Loan Status** button to run the ML model

### 📊 Main Dashboard — Prediction Results

- **Status Card** — Approved ✅ or Rejected ❌ with styled visual indicator
- **Probability Gauge** — Interactive Plotly gauge chart with color-coded risk bands (green/yellow/red)
- **Risk Level** — Low 🟢 / Medium 🟡 / High 🔴 classification
- **Feature Importance Chart** — Interactive multi-color Plotly bar chart showing top prediction factors
- **Model Performance Metrics** — Accuracy, precision, recall, F1 displayed in metric cards

### 🎨 Additional Features

- 🌗 **Theme Toggle** — Native dark/light mode switch with session-state persistence (predictions survive theme changes)
- 💬 **AI Chatbot** — Ask questions about predictions, eligibility criteria, loan guidelines, or application tips
- 📄 **PDF Download** — Generate and download a professional 8-page PDF report with:
  - Cover page with application ID
  - Executive summary with risk assessment
  - Full applicant profile (identity, personal, financial, loan sections)
  - AI decision report with detailed insights
  - Visual analytics and model comparison
  - Final summary with digital signature
- 📊 **Interactive Charts** — Plotly-powered charts with hover tooltips, zoom, and pan controls

---

## 🔮 Future Improvements

- [ ] 🗄️ **Database Integration** — Store predictions in SQLite or Firebase for historical tracking
- [ ] ☁️ **Cloud Deployment** — Deploy on Streamlit Cloud, Render, or AWS for public access
- [ ] 📱 **Responsive Mobile UI** — Optimize the dashboard layout for mobile devices
- [ ] 🔐 **User Authentication** — Add login/signup for secure access to predictions
- [ ] 📧 **Email Notifications** — Send PDF reports to applicants via email
- [ ] 🧠 **Advanced Models** — Integrate XGBoost, LightGBM, or Neural Networks for higher accuracy
- [ ] 📊 **Historical Analytics** — Track prediction trends and approval rates over time
- [ ] 🔄 **Model Retraining Pipeline** — Automated retraining with new data and performance monitoring

---

## 👨‍💻 Author

**Anees Ur Rehman**

- Developed as part of a **Data Science Internship** (Coursera)
- Built to demonstrate end-to-end ML project skills — from data analysis to deployment

---

<p align="center">
  <strong>⭐ Star this repo if you found it useful!</strong><br/>
  <em>Your support helps others discover this project.</em>
</p>

---

<p align="center">
  <sub>Built with ❤️ using Python, Scikit-Learn, Plotly & Streamlit</sub>
</p>

### Pipeline Steps

1. **Data Loading** — Load the loan dataset (`loan_dataset_2025.csv`) with applicant financial and personal attributes
2. **Data Cleaning** — Handle missing values using median (numeric) and mode (categorical) imputation
3. **Encoding** — Transform categorical features using `LabelEncoder` for model compatibility
4. **Feature Engineering** — Apply `log1p` transformations on `monthly_income` and `loan_amount` to reduce skewness
5. **Model Training** — Train three classifiers with optimized hyperparameters
6. **Model Evaluation** — Evaluate using accuracy, precision, recall, F1-score, ROC-AUC, and cross-validation
7. **Best Model Selection** — Automatically select the model with the highest accuracy score
8. **Deployment** — Serve predictions via an interactive Streamlit web application with real-time analytics

---

## 🧠 Model Training Details

Three classification models are trained and compared:

| Model | Configuration | Details |
|-------|--------------|---------|
| **Logistic Regression** | Default parameters | Linear baseline classifier for binary loan prediction |
| **Decision Tree** | `max_depth=8` | Non-linear classifier with controlled depth to prevent overfitting |
| **Random Forest** | `n_estimators=300` | Ensemble of 300 decision trees for robust predictions |

### Training Configuration

- **Train/Test Split:** 80/20 stratified split (preserves class distribution)
- **Encoding:** `LabelEncoder` for all categorical features
- **Feature Transforms:** `log1p` applied to `monthly_income` and `loan_amount`
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Cross-Validation:** K-fold cross-validation for model stability assessment
- **Artifact Storage:** Best model, feature list, and encoders saved to `models/loan_project_artifacts.pkl`

---

## 🖥️ UI Overview

The Streamlit web application provides a professional, fintech-grade interface:

*(Placeholder for UI Screenshot)*

### Main Dashboard

-   **KPI Cards**: At-a-glance view of the best model, its accuracy, dataset size, and session predictions.
-   **Prediction Results**: Displays the final decision (Approved/Rejected), probability score, risk level, and model confidence.
-   **Interactive Charts**: Includes a Plotly gauge for risk visualization and a bar chart for feature importance.
-   **Model Comparison**: A full dashboard comparing all trained models across multiple metrics, including ROC curves and confusion matrices.

### Sidebar

-   **Loan Application Form**: A comprehensive form for users to input applicant data.
-   **Theme Toggle**: A switch for seamless transition between dark and light modes.
-   **Dashboard Filters**: Controls to customize the model comparison view.

---

## ▶️ Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1.  **Clone the repository**
    ```bash
    git clone <https://github.com/your-username/Loan-Prediction-ML-App.git>
    cd Loan-Prediction-ML-App
    ```

2.  **Create and activate a virtual environment**
    ```bash
    # Create
    python -m venv venv

    # Activate (Windows)
    venv\Scripts\activate

    # Activate (macOS / Linux)
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

1.  **Train the ML models**
    First, run the training script. This will train all models, evaluate them, and save the best one and its artifacts to the `models/` directory.
    ```bash
    python train.py
    ```

2.  **Launch the Streamlit app**
    Once the training is complete, launch the web application.
    ```bash
    streamlit run app.py
    ```

The app will open in your browser at `http://localhost:8501`.

---

## 🚢 Deployment

This application is designed for easy deployment on Streamlit Cloud.

### Step 1: Push to GitHub

Ensure your project, including the `models/loan_project_artifacts.pkl` file, is pushed to a public GitHub repository.

### Step 2: Deploy on Streamlit Cloud

1.  Go to share.streamlit.io and sign in with GitHub.
2.  Click **"New app"** and select your repository.
3.  Set the **Main file path** to `app.py`.
4.  Click **"Deploy!"**.

Streamlit Cloud will automatically install the dependencies from `requirements.txt` and launch the application. Any `git push` to your main branch will trigger an automatic redeployment.

---

## 🔮 Future Improvements

- [ ] 🗄️ **Database Integration** — Store predictions in SQLite or Firebase for historical tracking
- [ ] ☁️ **Cloud Deployment** — Deploy on Streamlit Cloud, Render, or AWS for public access
- [ ] 📱 **Responsive Mobile UI** — Optimize the dashboard layout for mobile devices
- [ ] 🔐 **User Authentication** — Add login/signup for secure access to predictions
- [ ] 📧 **Email Notifications** — Send PDF reports to applicants via email
- [ ] 🧠 **Advanced Models** — Integrate XGBoost, LightGBM, or Neural Networks for higher accuracy
- [ ] 📊 **Historical Analytics** — Track prediction trends and approval rates over time
- [ ] 🔄 **Model Retraining Pipeline** — Automated retraining with new data and performance monitoring

---

## 🔗 Project Links

- **GitHub Repository:** https://github.com/your-username/project-name
- **Live Application:**[ ➡️ Access the Live Demo Here](https://loan-approval-prediction-system-ai.streamlit.app/)

---

## 👨‍💻 Author

**Anees Ur Rehman**

- Developed as part of a **Data Science Internship** (Coursera)
- Built to demonstrate end-to-end ML project skills — from data analysis to deployment

---

## 📜 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute with proper attribution.

© 2025 AI Loan Prediction System

---

<p align="center">
  <strong>⭐ Star this repo if you found it useful!</strong><br/>
  <em>Your support helps others discover this project.</em>
</p>

---

<p align="center">
  <sub>Built with ❤️ using Python, Scikit-Learn, Plotly & Streamlit</sub>
</p>
