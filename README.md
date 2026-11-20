# CivicFlow - AI Civic Complaint Classifier

CivicFlow is an Explainable AI (XAI)-powered civic technology web application that automatically classifies citizen complaints into five municipal departments: **Roads, Water, Electricity, Sanitation, and Traffic**.

Built using Machine Learning and Flask, CivicFlow helps streamline complaint management through intelligent classification, transparent prediction explanations, priority detection, and duplicate complaint identification.

The application features a clean, white **Material Design interface** with four main sections: Home, AI Insights, How Your Complaint Is Processed, and History.

---

## 🌟 Key Features

### 1. Explainable AI – Why This Prediction?
- **Model Attribution:** Explains predictions using TF-IDF feature values and Logistic Regression model coefficients.
- **Top Contributing Words:** Identifies and ranks influential words that contribute to the predicted department.
- **Visual Impact:** Displays the importance of contributing words through visual indicators and percentages.
- **Plain-Language Explanation:** Generates understandable explanations describing why a particular department was selected.
- **Fallback Handling:** Provides appropriate feedback when a complaint contains general or insufficiently specific wording.

### 2. Duplicate Complaint Detection
- **Cosine Similarity:** Compares incoming complaints against previously submitted complaints stored in the local database.
- **60% Similarity Threshold:** Identifies potentially similar complaints using a configurable similarity threshold.
- **Similar Complaint Details:** Displays the closest matching complaint along with its similarity percentage and submission date.
- **Responsible Labeling:** Uses the status "Similar Match Found" rather than automatically treating similar complaints as exact duplicates.

### 3. Urgency Detection
A rule-based priority detection system evaluates complaints independently of the machine learning classifier.

Three priority levels:
- 🔴 **High Priority:** Complaints containing emergency or potential hazard indicators.
- 🟡 **Medium Priority:** Complaints requiring attention but without immediate hazard indicators.
- 🟢 **Low Priority:** General complaints without identified urgency indicators.

Example hazard keywords include `emergency`, `burst`, `hazard`, `danger`, `flooding`, `sparking`, and `accident`.

### 4. Step-by-Step Complaint Processing
Displays the complaint preprocessing pipeline using individual step cards in a two-column layout.

| Step | Processing Stage |
|---|---|
| 01 | Original Complaint Received |
| 02 | Convert to Lowercase |
| 03 | Remove Punctuation and Symbols |
| 04 | Clean Extra Spaces |
| 05 | Identify Department Keywords |

The interface also provides a **Before & After Comparison** to show how the original complaint is transformed into normalized text.

### 5. Human Review Alert
A confidence-based guardrail evaluates the model's prediction probability against a configurable 60% threshold.

- **Below 60%:** Displays a low-confidence warning recommending staff verification.
- **60% and Above:** Indicates that prediction confidence is sufficient for automatic routing.

This feature helps identify uncertain predictions that may require human attention.

### 6. Complaint History & Details
- **SQLite Persistence:** Stores complaints, predicted departments, confidence scores, priority levels, and timestamps.
- **Compact History List:** Displays complaint previews, status badges, dates, and confidence scores.
- **Detailed Complaint View:** Opens the complete saved complaint record when a history entry is selected.
- **Database Retrieval:** Retrieves the original saved prediction without running the classifier again.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Backend | Python, Flask |
| Machine Learning | Scikit-learn, Logistic Regression |
| Text Processing | TF-IDF Vectorization |
| Explainable AI | Logistic Regression Feature Attribution |
| Similarity Detection | Cosine Similarity |
| Database | SQLite |
| Frontend | HTML, CSS, JavaScript |
| UI Design | Material Design |
| Testing | Python Unit Testing |

---

## 🏗️ Project Architecture

```text
CivicFlow/
│
├── dataset/
│   └── complaints.csv
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── database.py
├── train_model.py
├── test_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Main Components

| File | Description |
|---|---|
| `app.py` | Flask application, prediction API, XAI logic, urgency detection, and duplicate detection. |
| `database.py` | SQLite database initialization, complaint storage, and history retrieval. |
| `train_model.py` | Trains the TF-IDF vectorizer and Logistic Regression classifier. |
| `test_app.py` | Automated tests covering application features and API endpoints. |
| `complaints.csv` | Dataset containing sample civic complaints and their department labels. |
| `model.pkl` | Trained Logistic Regression model. |
| `vectorizer.pkl` | Fitted TF-IDF vectorizer. |

---

## 🚀 Installation and Setup

### Prerequisites

Make sure the following are installed:

- Python 3.10 or above
- Git
- A web browser

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/CivicFlow.git
cd CivicFlow
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

If script execution is disabled, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Train the Machine Learning Model (Optional)

Pre-trained model files are included in the repository.

To retrain the model using the dataset:

```bash
python train_model.py
```

### Step 6: Run the Application

```bash
python app.py
```

The Flask development server starts on port `5000` and initializes the SQLite database.

### Step 7: Open CivicFlow

Navigate to the following address in your browser:

```text
http://localhost:5000
```

---

## 🧪 Running Automated Tests

To execute the automated test suite:

```bash
python test_app.py
```

The test suite covers:

1. Classification across all five municipal departments.
2. Low-confidence prediction and human review alerts.
3. Feature attribution and influential word extraction.
4. Text preprocessing pipeline.
5. Duplicate complaint detection.
6. SQLite database initialization and record operations.
7. History API and individual complaint retrieval.
8. Empty input validation and error handling.
9. Frontend template rendering.

---

## 📋 Sample Test Scenarios

| Test Scenario | Sample Input / Action | Expected Result |
|---|---|---|
| Roads Classification | Select Roads Issue template. | Roads department prediction. |
| Water Classification | Select Water Leakage template. | Water department prediction. |
| Electricity Classification | Select Electricity Issue template. | Electricity department prediction. |
| Urgency Detection | "Emergency! Water pipe burst flooding the road!" | High Priority. |
| Duplicate Detection | Submit the same complaint twice. | Similar Match Found. |
| Low Confidence | "The general public area seems somewhat dark." | Human review recommendation. |
| Processing Pipeline | Open How Your Complaint Is Processed. | Five individual preprocessing cards. |
| Complaint History | Open History and select a record. | Complete previously saved complaint details. |

---

## 🔍 How CivicFlow Works

The application follows a structured complaint-processing workflow:

1. **Complaint Submission:** A citizen enters a municipal complaint through the Home section.
2. **Text Preprocessing:** The complaint undergoes normalization and cleaning.
3. **Department Classification:** The TF-IDF vectorizer transforms the text into numerical features, which are processed by the trained Logistic Regression model.
4. **Explainable Prediction:** Feature attribution identifies influential words behind the predicted department.
5. **Priority Assessment:** A separate rule-based system determines the complaint's urgency.
6. **Duplicate Detection:** Cosine similarity compares the complaint against existing records.
7. **Confidence Evaluation:** The system determines whether the prediction meets the configured confidence threshold.
8. **Database Storage:** The complaint and prediction details are stored in SQLite for future reference.

---

## 📌 Project Objectives

- Automate the classification of municipal complaints.
- Improve transparency through Explainable AI.
- Identify potentially urgent civic issues.
- Reduce repeated complaint submissions through similarity detection.
- Support human verification for uncertain predictions.
- Maintain accessible complaint history and records.

---
## 👨‍💻 Developer

**Revanasidda Nimbal** is the developer of CivicFlow.

- GitHub: [**RevanasiddaNimbal**](https://github.com/RevanasiddaNimbal)
- LinkedIn: [**Revanasidda Nimbal**](https://www.linkedin.com/in/revanasidda-nimbal-68a596365/)
- Email: `revanasiddanimbal82@gmail.com`
## 📄 License

This project is developed as an educational and civic technology application focused on Explainable AI and intelligent complaint management.

