# HealthInsight – Healthcare Data Analytics & Patient Risk Prediction Platform

> **Healthcare Data Analytics Internship Project**  
> *A full-stack clinical analytics and machine-learning risk prediction platform for inpatient cohort management, data preprocessing, and exploratory healthcare intelligence.*

---

## 1. Project Overview

**HealthInsight** bridges the gap between raw electronic health record (EHR) datasets and actionable clinical insights. Modern healthcare institutions process vast volumes of patient admissions, diagnoses, and lab values, yet raw data is often noisy, incomplete, and difficult to interpret without dedicated analytical pipelines.

This platform provides:
1. **Secure JWT Authentication**: Role-based access for clinical analysts and researchers with bcrypt password hashing.
2. **Automated Data Cleaning Pipeline**: Missing value imputation (median/mean/mode), duplicate detection, and Interquartile Range (IQR) outlier clipping with before/after audit summaries.
3. **Exploratory Data Analysis (EDA)**: Descriptive statistics (mean, median, min, max, standard deviation, quartiles), Pearson correlation matrix, and an interactive dynamic visualization studio.
4. **Machine Learning Risk Prediction**: Supervised Scikit-learn classification models (Random Forest, Decision Tree, Logistic Regression) with 80/20 train/test evaluation, confusion matrices, and real-time patient risk inference.
5. **Interactive Executive Dashboard**: 8 real-time KPI cards and 10 dynamic charts responding reactively to demographic and diagnostic filters.
6. **Executive Clinical Reporting**: One-click dossier generation with downloadable vector PDF export.

---

## 2. Technology Stack

### Frontend
- **Framework**: React 19 (TypeScript)
- **Build Tool**: Vite
- **Styling**: Tailwind CSS (Healthcare SaaS aesthetic, clean whites, teal & cyan accents, zero-pill metadata)
- **Routing**: React Router v7
- **Data Visualization**: Recharts & Lucide React
- **PDF Generation**: jsPDF

### Backend & API
- **Live Interactive Engine**: Node.js & Express with full REST API endpoints, JWT authentication, and atomic JSON/SQLite persistent storage.
- **Python Production Backend**: Python 3.10+, FastAPI, SQLAlchemy, Pydantic, Uvicorn, Pandas, NumPy, Scikit-learn.
- **Database**: PostgreSQL (with SQLite fallback for local developer workstations).

---

## 3. Sample Demo Credentials

For immediate evaluation, the application is pre-seeded with an active synthetic cohort of **1,200 patient records** and demo credentials:

- **Email**: `intern@healthinsight.org`
- **Password**: `Password123!`
- **Role**: Clinical Data Analyst Intern

*(You can also register a new account on the `/register` page with real-time password strength validation).*

---

## 4. Architecture & Directory Structure

```
healthinsight/
├── data/
│   ├── healthinsight.json          # Persistent database store
│   └── sample_healthcare_data.csv  # 1,200 synthetic patient records
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── database.py             # SQLAlchemy engine & session management
│   │   ├── models.py               # ORM database models
│   │   └── schemas.py              # Pydantic validation schemas
│   └── requirements.txt            # Python dependencies
├── src/
│   ├── assets/                     # Clinical UI imagery & vectors
│   ├── components/
│   │   ├── common/                 # ProtectedRoute, Toast, Loaders
│   │   └── layout/                 # Top Navbar (3-zone), Sidebar, AppLayout
│   ├── context/
│   │   ├── AuthContext.tsx         # JWT state & persistent session hook
│   │   └── ToastContext.tsx        # Non-intrusive alert notification provider
│   ├── pages/
│   │   ├── auth/                   # Login, Register, Forgot & Reset Password
│   │   ├── LandingPage.tsx         # Public product landing page
│   │   ├── DashboardPage.tsx       # 8 KPI cards & 10 interactive charts
│   │   ├── PatientsPage.tsx        # Patient table, CRUD modals, CSV export
│   │   ├── DatasetsPage.tsx        # Drag & drop upload, IQR cleaning pipeline
│   │   ├── AnalyticsPage.tsx       # Parametric EDA & Pearson correlation matrix
│   │   ├── RiskPredictionPage.tsx  # Scikit-learn ML training & Risk Calculator
│   │   ├── ReportsPage.tsx         # Clinical report preview & PDF exporter
│   │   ├── ProfilePage.tsx         # User profile & credentials management
│   │   └── ProjectPage.tsx         # 6-Week Internship timeline & curriculum
│   ├── services/
│   │   └── api.ts                  # Typed HTTP client with Bearer tokens
│   ├── types/
│   │   └── index.ts                # TypeScript domain models
│   ├── App.tsx                     # React Router definition
│   └── index.css                   # Tailwind styles & print stylesheet
├── server/
│   ├── analyticsEngine.ts          # Descriptive statistics & correlation math
│   ├── cleaner.ts                  # Missing value, duplicate, and IQR pipeline
│   ├── dataGenerator.ts            # 1,200 synthetic clinical records generator
│   ├── db.ts                       # Database abstraction & initial seeding
│   └── mlEngine.ts                 # Classification algorithms & risk scoring
├── .env.example
├── docker-compose.yml
├── package.json
├── server.ts                       # Full-stack server entry point (port 3000)
└── README.md
```

---

## 5. Quickstart & Setup Commands

### Running in Current Environment
```bash
# Start full-stack application (frontend + live API server on port 3000)
npm run dev
```

### Running Locally on Windows (PowerShell / CMD)

#### Step 1: Clone and Install Node.js Frontend & Server
```powershell
# Navigate into project directory
cd healthinsight

# Install Node dependencies
npm install

# Start the full-stack server on port 3000
npm run dev
```
Open your browser to `http://localhost:3000`.

#### Step 2 (Optional): Running Python FastAPI Backend
```powershell
# Create Python virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1
# Or in Windows Command Prompt:
# .\venv\Scripts\activate.bat

# Install Python requirements
pip install -r backend/requirements.txt

# Run FastAPI server with Uvicorn
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Running with Docker Compose
```bash
docker-compose up --build
```

---

## 6. Environment Variables (`.env.example`)

```ini
PORT=3000
NODE_ENV=development
JWT_SECRET="healthinsight_jwt_secret_key_production_2026"
DATABASE_URL="sqlite:///./data/healthinsight.db"
# For PostgreSQL:
# DATABASE_URL="postgresql://health_user:health_password@localhost:5432/healthinsight"
```

---

## 7. Machine Learning Methodology

The machine learning subsystem implements supervised classification pipelines:
1. **Pre-processing**: Continuous biomarkers (age, BMI, systolic BP, diastolic BP, glucose, cholesterol, heart rate) are scaled and normalized.
2. **Train/Test Partition**: Data is deterministically split into 80% training and 20% hold-out test sets.
3. **Classifiers**:
   - **Random Forest**: Ensemble bootstrap aggregation measuring Gini impurity reduction.
   - **Decision Tree**: Greedy recursive partition based on entropy/Gini gain.
   - **Logistic Regression**: Linear log-odds decision boundary with L2 regularization.
4. **Validation Metrics**: Accuracy, Precision, Recall, F1-Score, and a complete Confusion Matrix.
5. **Contributing Factor Attribution**: When calculating patient risk, individual vital elevations (e.g., Systolic BP $\ge$ 145 mmHg, Glucose $\ge$ 140 mg/dL) are weighted to provide transparent, explainable recommendations.

---

## 8. 6-Week Internship Curriculum Mapping

| Milestone | Phase | Description |
| :--- | :--- | :--- |
| **Week 1** | Strategic Planning & Scope | Hospital readmissions problem definition, EHR privacy compliance review, system architecture. |
| **Week 2** | Data Collection & Preparation | 1,200+ record synthetic cohort generation, CSV parsing, IQR outlier handling, missing value imputation. |
| **Week 3** | Exploratory Data Analysis | Parametric summary statistics, Pearson correlation coefficient matrix (r), multi-axis visualization studio. |
| **Week 4** | Machine Learning & Risk Modeling | Scikit-learn classification training, 80/20 train/test evaluation, real-time risk calculator. |
| **Week 5** | Dashboard & Clinical UI | 8 KPI metrics, 10 dynamic charts, multi-parameter filtering, patient management CRUD table. |
| **Week 6** | Reporting & Defense Dossier | Automated clinical report generation, vector PDF export, future EHR interoperability (FHIR) documentation. |

---

## 9. Non-Diagnostic Disclaimer

HealthInsight is engineered for **educational, academic research, and healthcare data analytics demonstrations**. All predictive outputs, risk categories, and correlation calculations are derived from statistical algorithms and must never replace clinical diagnosis, bedside medical judgment, or the counsel of licensed medical practitioners.
