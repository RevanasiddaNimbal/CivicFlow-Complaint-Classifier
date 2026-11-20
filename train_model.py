import os
import re
import string
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Comprehensive, domain-specific whitelisted civic vocabulary
CIVIC_VOCAB = [
    # Electricity
    "electricity", "electric", "power", "wire", "wires", "wiring", "cable", "cables", 
    "transformer", "transformers", "blackout", "blackouts", "voltage", "fluctuation", "fluctuations", 
    "lamp", "lamps", "bulb", "bulbs", "sparking", "spark", "sparks", "current", "shock", "circuit", 
    "electrocution", "pole", "poles", "generator", "generators", "fuse", "fuses", "powercut", "powercuts", 
    "outage", "outages", "light", "lights",
    # Roads
    "road", "roads", "pothole", "potholes", "pavement", "pavements", "asphalt", "tar", "sinkhole", "sinkholes", "trench", 
    "trenches", "manhole", "manholes", "flyover", "flyovers", "bridge", "bridges", "lane", "lanes", 
    "cracks", "cracking", "uneven", "bump", "bumps", "speedbreaker", "speedbreakers", "sidewalk", "sidewalks", "surface",
    # Sanitation
    "garbage", "trash", "rubbish", "litter", "waste", "bin", "bins", "dump", "dumping", "landfill", 
    "sweep", "swept", "unswept", "uncollected", "sewer", "sewers", "sewage", "drain", "drains", "drainage", 
    "gutter", "gutters", "odor", "smell", "foul", "stink", "mosquito", "mosquitoes", "hygiene", 
    "unhygienic", "clean", "cleaned", "cleaning", "sweepers", "dumpster", "dumpsters", "flies", "rot", "rotting",
    # Traffic
    "traffic", "signal", "signals", "sign", "signs", "signage", "vehicle", "vehicles", "car", "cars", 
    "truck", "trucks", "parking", "parked", "gridlock", "congestion", "jam", "jams", "queue", "queues", 
    "bottleneck", "bottlenecks", "speeding", "speed", "wrongway", "commuter", "commuters", "police", 
    "warden", "doubleparked", "block", "blocked", "blocking", "blockage", "delay", "delays",
    # Water
    "water", "pipe", "pipes", "pipeline", "pipelines", "leak", "leaks", "leakage", "leakages", 
    "valve", "valves", "burst", "pressure", "contaminated", "contamination", "tap", "taps", 
    "supply", "flooding", "flood", "floods", "pump", "pumping"
]
CIVIC_VOCAB = list(set([w.lower() for w in CIVIC_VOCAB]))

def clean_text(text):
    """
    Cleans complaint text by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Removing extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def train_classifier():
    dataset_path = os.path.join("dataset", "complaints.csv")
    models_dir = "models"
    
    print(f"Loading dataset from {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
        
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded. Total records: {len(df)}")
    
    # Check for missing values
    print("Checking for missing values...")
    print(df.isnull().sum())
    df = df.dropna(subset=["complaint", "department"])
    
    # Preprocess text
    print("Cleaning complaint text...")
    df["clean_complaint"] = df["complaint"].apply(clean_text)
    
    # Display simple statistics of departments
    print("\nDepartment Distribution:")
    print(df["department"].value_counts())
    
    X = df["clean_complaint"]
    y = df["department"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Feature Extraction (TF-IDF Vectorization)
    print("\nExtracting features using TF-IDF Vectorizer with whitelisted vocabulary...")
    vectorizer_eval = TfidfVectorizer(vocabulary=CIVIC_VOCAB)
    X_train_vectorized = vectorizer_eval.fit_transform(X_train)
    X_test_vectorized = vectorizer_eval.transform(X_test)
    
    # Model Training for Evaluation
    print("Training Logistic Regression model for evaluation (fit_intercept=False)...")
    model_eval = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=False)
    model_eval.fit(X_train_vectorized, y_train)
    
    # Evaluation
    predictions = model_eval.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Evaluation Results:")
    print(f"Accuracy Score: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    # Retrain on the ENTIRE dataset
    print("\nRetraining final model on the entire dataset (100% data) for deployment...")
    vectorizer = TfidfVectorizer(vocabulary=CIVIC_VOCAB)
    X_full_vectorized = vectorizer.fit_transform(X)
    
    model = LogisticRegression(random_state=42, max_iter=1000, fit_intercept=False)
    model.fit(X_full_vectorized, y)
    
    # Save the models, to avoid retraining on every run
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")
        
    model_path = os.path.join(models_dir, "model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    
    print(f"Saving final model to {model_path}...")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
        
    print(f"Saving final vectorizer to {vectorizer_path}...")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("Training process finished successfully!")

if __name__ == "__main__":
    train_classifier()
