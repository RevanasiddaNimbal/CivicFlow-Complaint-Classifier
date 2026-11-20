import os
import re
import string
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import database

app = Flask(__name__)

# Model and vectorizer file paths
MODEL_PATH = os.path.join("models", "model.pkl")
VECTORIZER_PATH = os.path.join("models", "vectorizer.pkl")

# Global variables for model and vectorizer
model = None
vectorizer = None

# Configurable thresholds
DUPLICATE_SIMILARITY_THRESHOLD = 0.60  # 60% cosine similarity threshold
CONFIDENCE_REVIEW_THRESHOLD = 0.60     # 60% prediction confidence threshold

# Domain keywords for urgency detection (Rule-based)
HIGH_PRIORITY_KEYWORDS = [
    "emergency", "dangerous", "danger", "hazard", "hazardous",
    "accident", "severe", "critical", "immediate", "immediately",
    "sparking", "spark", "sparks", "fire", "burst", "bursting",
    "electrocution", "shock", "collapsed", "sinkhole", "sinkholes",
    "flooding", "flood", "urgent", "urgently", "ambulance", "casualty",
    "injury", "injuries", "life-threatening", "explosion", "exploding"
]

MEDIUM_PRIORITY_KEYWORDS = [
    "blocked", "block", "blocking", "blockage", "leak", "leaks",
    "leakage", "overflowing", "overflow", "broken", "blackout",
    "blackouts", "powercut", "powercuts", "outage", "outages",
    "flickering", "damaged", "damage", "stink", "odor", "foul",
    "smell", "crack", "cracks", "pothole", "potholes", "gridlock",
    "congestion", "jam", "delay", "delays", "contaminated",
    "contamination", "unhygienic", "flies", "mosquitoes", "pressure"
]

LOW_PRIORITY_KEYWORDS = [
    "minor", "routine", "aesthetic", "paint", "painting", "sweep",
    "swept", "request", "inquiry", "suggestion", "slow", "notice"
]


def clean_text(text):
    """
    Cleans complaint text in the exact same way as during training:
    1. Lowercasing
    2. Punctuation removal
    3. Whitespace normalization
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_models():
    """Loads the trained Logistic Regression model and TF-IDF vectorizer."""
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print("Models loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def get_preprocessing_preview(raw_text):
    """
    Demonstrates the step-by-step preprocessing pipeline on raw complaint text.
    Reuses the exact transformation steps of clean_text.
    """
    if not isinstance(raw_text, str):
        raw_text = ""

    step1_raw = raw_text
    step2_lower = raw_text.lower()
    step3_nopunct = step2_lower.translate(str.maketrans("", "", string.punctuation))
    step4_clean = re.sub(r"\s+", " ", step3_nopunct).strip()

    # Identify vocabulary tokens present in the cleaned text
    matched_vocab_tokens = []
    if vectorizer is not None:
        try:
            vocab_set = set(vectorizer.get_feature_names_out())
            matched_vocab_tokens = [w for w in step4_clean.split() if w in vocab_set]
            # Deduplicate preserving order
            matched_vocab_tokens = list(dict.fromkeys(matched_vocab_tokens))
        except Exception:
            matched_vocab_tokens = []

    steps = [
        {
            "step_number": 1,
            "icon": "📝",
            "title": "Original Complaint Received",
            "description": "We start with the exact text you typed, keeping your original capital letters, punctuation, and spacing.",
            "output": step1_raw
        },
        {
            "step_number": 2,
            "icon": "🔤",
            "title": "Convert to Lowercase",
            "description": "All letters are changed to lowercase so words like 'Pothole' and 'pothole' are read as the exact same word.",
            "output": step2_lower
        },
        {
            "step_number": 3,
            "icon": "✂️",
            "title": "Remove Punctuation and Symbols",
            "description": "Marks like commas, periods, and exclamation points are removed so the model can focus cleanly on each word.",
            "output": step3_nopunct
        },
        {
            "step_number": 4,
            "icon": "🧹",
            "title": "Clean Extra Spaces",
            "description": "Extra spaces, tabs, and line breaks are cleaned up to make the text neat and consistent.",
            "output": step4_clean
        },
        {
            "step_number": 5,
            "icon": "🎯",
            "title": "Find Department Keywords",
            "description": "The system searches for important civic words matching municipal departments, such as potholes, water, wiring, garbage, or traffic.",
            "output": ", ".join(matched_vocab_tokens) if matched_vocab_tokens else "No specific department keywords found"
        }
    ]

    return {
        "original": step1_raw,
        "cleaned": step4_clean,
        "matched_vocab": matched_vocab_tokens,
        "steps": steps
    }


def detect_urgency(text):
    """
    Feature 3: Rule-based Urgency Detection.
    Evaluates urgency indicators (emergency, danger, severe, minor, etc.).
    Keeps urgency rule-based and completely separate from ML department classification.
    """
    lower = text.lower()

    # Check High Priority keywords
    matched_high = [
        kw for kw in HIGH_PRIORITY_KEYWORDS 
        if re.search(r'\b' + re.escape(kw) + r'\b', lower)
    ]
    if matched_high:
        return {
            "level": "High Priority",
            "badge_class": "badge-priority-high",
            "matched_keywords": matched_high,
            "reason": f"Urgent safety hazard keyword detected: {', '.join(matched_high[:3])}",
            "description": "Immediate public safety or critical infrastructure risk detected. Prioritize rapid field response.",
            "is_rule_based": True
        }

    # Check Medium Priority keywords
    matched_med = [
        kw for kw in MEDIUM_PRIORITY_KEYWORDS 
        if re.search(r'\b' + re.escape(kw) + r'\b', lower)
    ]
    if matched_med:
        return {
            "level": "Medium Priority",
            "badge_class": "badge-priority-medium",
            "matched_keywords": matched_med,
            "reason": f"Active disruption indicator detected: {', '.join(matched_med[:3])}",
            "description": "Significant civic disruption or service degradation. Scheduled repair and maintenance needed.",
            "is_rule_based": True
        }

    # Low Priority (Routine or default)
    matched_low = [
        kw for kw in LOW_PRIORITY_KEYWORDS 
        if re.search(r'\b' + re.escape(kw) + r'\b', lower)
    ]
    return {
        "level": "Low Priority",
        "badge_class": "badge-priority-low",
        "matched_keywords": matched_low if matched_low else ["Routine civic report"],
        "reason": "Standard non-urgent civic maintenance or observation.",
        "description": "Non-acute civic issue. Can be handled under standard department queuing.",
        "is_rule_based": True
    }


def explain_prediction(cleaned_complaint, predicted_dept, model_obj, vectorizer_obj):
    """
    Feature 1: Explainable AI – Why This Prediction?
    Calculates feature contribution using actual Logistic Regression coefficients
    and TF-IDF weights: Contribution = TF-IDF * Coefficient.
    Identifies the most influential words driving the prediction.
    """
    if model_obj is None or vectorizer_obj is None:
        return {
            "contributing_words": [],
            "explanation": "Model objects not initialized.",
            "has_contributing_words": False
        }

    try:
        feature_names = vectorizer_obj.get_feature_names_out()
        classes_list = list(model_obj.classes_)
        if predicted_dept not in classes_list:
            return {
                "contributing_words": [],
                "explanation": f"Department {predicted_dept} not found in model classes.",
                "has_contributing_words": False
            }

        class_idx = classes_list.index(predicted_dept)
        weights = model_obj.coef_[class_idx]

        # Vectorize the cleaned complaint
        tfidf_vec = vectorizer_obj.transform([cleaned_complaint]).toarray()[0]

        word_contributions = []
        for col_idx, tfidf_val in enumerate(tfidf_vec):
            if tfidf_val > 0:
                coef = float(weights[col_idx])
                contrib = float(tfidf_val * coef)
                word_contributions.append({
                    "word": feature_names[col_idx],
                    "weight": round(coef, 4),
                    "tfidf": round(float(tfidf_val), 4),
                    "contribution": round(contrib, 4)
                })

        # Filter strictly for positive contributors that pushed towards this department
        positive_contributors = [w for w in word_contributions if w["contribution"] > 0]
        positive_contributors.sort(key=lambda x: x["contribution"], reverse=True)

        if positive_contributors:
            total_positive = sum(w["contribution"] for w in positive_contributors)
            for w in positive_contributors:
                w["impact_percentage"] = (
                    round((w["contribution"] / total_positive) * 100, 1)
                    if total_positive > 0 else 0.0
                )
                if w["impact_percentage"] >= 50:
                    w["importance_level"] = "High Importance"
                elif w["impact_percentage"] >= 25:
                    w["importance_level"] = "Medium Importance"
                else:
                    w["importance_level"] = "Supporting Keyword"

            top_words = [f"'{w['word']}'" for w in positive_contributors[:3]]
            if len(top_words) == 1:
                words_phrase = top_words[0]
            elif len(top_words) == 2:
                words_phrase = f"{top_words[0]} and {top_words[1]}"
            else:
                words_phrase = f"{', '.join(top_words[:-1])}, and {top_words[-1]}"

            explanation = (
                f"The words related to {words_phrase} contributed most strongly to the "
                f"{predicted_dept} department prediction."
            )
            has_contributing_words = True
        else:
            explanation = (
                f"No specific civic keywords strongly influenced this prediction. "
                f"The model assigned the {predicted_dept} department based on general phrasing."
            )
            has_contributing_words = False

        return {
            "contributing_words": positive_contributors[:6],  # Return top influential words
            "explanation": explanation,
            "has_contributing_words": has_contributing_words
        }

    except Exception as e:
        print(f"Error in explain_prediction: {e}")
        return {
            "contributing_words": [],
            "explanation": "Could not compute feature attribution due to an internal error.",
            "has_contributing_words": False
        }


def check_duplicate_complaint(cleaned_complaint, stored_complaints, threshold=DUPLICATE_SIMILARITY_THRESHOLD):
    """
    Feature 2: Duplicate Complaint Detection.
    Compares the newly submitted complaint against previously stored complaints
    using TF-IDF vectorization and cosine similarity.
    """
    if not stored_complaints or not cleaned_complaint.strip():
        return {
            "is_similar": False,
            "similarity_percentage": 0.0,
            "threshold_percentage": round(threshold * 100, 1),
            "similar_complaint": None,
            "similar_department": None,
            "similar_date": None,
            "message": "No previously stored complaints available for comparison."
        }

    try:
        stored_texts = [item["cleaned_text"] for item in stored_complaints]
        all_texts = [cleaned_complaint] + stored_texts

        sim_vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_mat = sim_vectorizer.fit_transform(all_texts)
        except ValueError:
            sim_vectorizer = TfidfVectorizer()
            tfidf_mat = sim_vectorizer.fit_transform(all_texts)

        # Compute cosine similarity between new complaint (index 0) and stored complaints (index 1:)
        sim_scores = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:])[0]
        best_idx = int(np.argmax(sim_scores))
        best_score = float(sim_scores[best_idx])
        sim_percentage = round(best_score * 100, 1)

        if best_score >= threshold:
            match = stored_complaints[best_idx]
            return {
                "is_similar": True,
                "similarity_percentage": sim_percentage,
                "threshold_percentage": round(threshold * 100, 1),
                "similar_complaint": match["original_text"],
                "similar_department": match["predicted_department"],
                "similar_date": match["created_at"],
                "message": "Potentially similar complaint detected (not a guaranteed duplicate)."
            }
        else:
            return {
                "is_similar": False,
                "similarity_percentage": sim_percentage,
                "threshold_percentage": round(threshold * 100, 1),
                "similar_complaint": None,
                "similar_department": None,
                "similar_date": None,
                "message": "No similar complaints found."
            }

    except Exception as e:
        print(f"Error checking duplicate complaint: {e}")
        return {
            "is_similar": False,
            "similarity_percentage": 0.0,
            "threshold_percentage": round(threshold * 100, 1),
            "similar_complaint": None,
            "similar_department": None,
            "similar_date": None,
            "message": "Could not execute similarity check."
        }


def check_human_review(confidence, threshold=CONFIDENCE_REVIEW_THRESHOLD):
    """
    Feature 5: Human Review Alert.
    Issues a confidence-based warning if model confidence is below the threshold.
    """
    conf_pct = round(confidence * 100, 1)
    thresh_pct = round(threshold * 100, 1)

    if confidence < threshold:
        return {
            "needs_review": True,
            "alert_type": "warning",
            "badge_class": "badge-review-warning",
            "message": "Low confidence - Human verification recommended.",
            "detail": f"Routing confidence is {conf_pct}%, which is below the {thresh_pct}% automatic verification threshold. Phrasing may be ambiguous.",
            "threshold_percentage": thresh_pct
        }
    else:
        return {
            "needs_review": False,
            "alert_type": "success",
            "badge_class": "badge-review-success",
            "message": "Prediction confidence is sufficient for automatic classification.",
            "detail": f"Routing confidence is {conf_pct}%, meeting the {thresh_pct}% automatic verification standard.",
            "threshold_percentage": thresh_pct
        }


# =====================================================================
# Flask Routes
# =====================================================================

@app.route("/")
def home():
    """Renders the main single-page interface."""
    models_ready = True
    if model is None or vectorizer is None:
        models_ready = load_models()
    return render_template("index.html", models_ready=models_ready)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Primary prediction and analysis endpoint.
    Executes department prediction and all six AI/NLP features.
    """
    global model, vectorizer
    if model is None or vectorizer is None:
        if not load_models():
            return jsonify({
                "status": "error",
                "error": "Model files are missing. Please train the model first by running train_model.py."
            }), 500

    data = request.get_json()
    if not data or "complaint" not in data:
        return jsonify({
            "status": "error",
            "error": "No complaint description provided."
        }), 400

    complaint_text = data["complaint"]
    if not isinstance(complaint_text, str) or not complaint_text.strip():
        return jsonify({
            "status": "error",
            "error": "Complaint text cannot be empty."
        }), 400

    # Feature 4: Preprocessing preview & clean text
    preprocessing_info = get_preprocessing_preview(complaint_text)
    cleaned_complaint = preprocessing_info["cleaned"]

    # Feature 3: Urgency detection
    urgency_info = detect_urgency(complaint_text)

    # Feature 2: Duplicate complaint detection (against prior stored complaints)
    previous_complaints = database.get_all_complaints_for_similarity()
    duplicate_info = check_duplicate_complaint(
        cleaned_complaint,
        previous_complaints,
        threshold=DUPLICATE_SIMILARITY_THRESHOLD
    )

    # Machine Learning Prediction
    vectorized_text = vectorizer.transform([cleaned_complaint])
    predicted_dept = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    class_labels = model.classes_

    confidence_scores = {
        label: float(prob) for label, prob in zip(class_labels, probabilities)
    }
    sorted_confidence = dict(
        sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    )
    primary_confidence = float(confidence_scores[predicted_dept])

    # Feature 1: Explainable AI
    explainability_info = explain_prediction(
        cleaned_complaint,
        predicted_dept,
        model,
        vectorizer
    )

    # Feature 5: Human Review Alert
    human_review_info = check_human_review(
        primary_confidence,
        threshold=CONFIDENCE_REVIEW_THRESHOLD
    )

    # Feature 6: Store record in SQLite History Database
    details_payload = {
        "complaint": complaint_text,
        "clean_complaint": cleaned_complaint,
        "prediction": predicted_dept,
        "confidence": primary_confidence,
        "all_probabilities": sorted_confidence,
        "explainability": explainability_info,
        "duplicate_detection": duplicate_info,
        "urgency": urgency_info,
        "preprocessing": preprocessing_info,
        "human_review": human_review_info
    }

    try:
        database.insert_prediction(
            original_text=complaint_text,
            cleaned_text=cleaned_complaint,
            predicted_dept=predicted_dept,
            confidence=primary_confidence,
            urgency=urgency_info["level"],
            urgency_reason=urgency_info["reason"],
            details=details_payload
        )
    except Exception as e:
        print(f"Warning: Could not save complaint to history: {e}")

    return jsonify({
        "status": "success",
        "complaint": complaint_text,
        "clean_complaint": cleaned_complaint,
        "prediction": predicted_dept,
        "confidence": primary_confidence,
        "all_probabilities": sorted_confidence,
        "explainability": explainability_info,
        "duplicate_detection": duplicate_info,
        "urgency": urgency_info,
        "preprocessing": preprocessing_info,
        "human_review": human_review_info
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns recently classified complaints from the database."""
    try:
        records = database.get_history(limit=50)
        return jsonify({
            "status": "success",
            "count": len(records),
            "history": records
        })
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({
            "status": "error",
            "error": "Failed to retrieve history."
        }), 500


@app.route("/api/history/<int:record_id>", methods=["GET"])
def get_history_detail(record_id):
    """
    Returns the saved prediction details of a specific history record directly from the database.
    Does NOT run a new prediction.
    """
    try:
        import json
        record = database.get_prediction_by_id(record_id)
        if not record:
            return jsonify({
                "status": "error",
                "error": "Complaint record not found."
            }), 404

        details = None
        if record.get("details_json"):
            try:
                details = json.loads(record["details_json"])
            except Exception as e:
                print(f"Error parsing details_json: {e}")
                details = None

        if not details:
            details = {
                "complaint": record["original_text"],
                "clean_complaint": record["cleaned_text"],
                "prediction": record["predicted_department"],
                "confidence": record["confidence"],
                "all_probabilities": {record["predicted_department"]: record["confidence"]},
                "explainability": {
                    "explanation": f"The model selected the {record['predicted_department']} department based on complaint keywords.",
                    "contributing_words": [],
                    "has_contributing_words": False
                },
                "urgency": {
                    "level": record["urgency"],
                    "reason": record["urgency_reason"] or "Priority determined at submission time.",
                    "matched_keywords": []
                },
                "duplicate_detection": {
                    "is_similar": False,
                    "similarity_percentage": 0.0,
                    "message": "No similarity data recorded."
                },
                "human_review": {
                    "needs_review": record["confidence"] < 0.60,
                    "message": "Review recommended" if record["confidence"] < 0.60 else "Automatic routing verified",
                    "detail": "Saved from previous classification."
                },
                "preprocessing": {
                    "original": record["original_text"],
                    "cleaned": record["cleaned_text"],
                    "steps": []
                }
            }

        return jsonify({
            "status": "success",
            "record_id": record["id"],
            "created_at": record["created_at"],
            "details": details
        })
    except Exception as e:
        print(f"Error retrieving history detail: {e}")
        return jsonify({
            "status": "error",
            "error": "Failed to retrieve complaint details."
        }), 500


@app.route("/api/history/clear", methods=["POST"])
def clear_history_route():
    """Clears all stored prediction history."""
    try:
        database.clear_history()
        return jsonify({
            "status": "success",
            "message": "Prediction history cleared successfully."
        })
    except Exception as e:
        print(f"Error clearing history: {e}")
        return jsonify({
            "status": "error",
            "error": "Failed to clear history."
        }), 500


# Fallback error handlers
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"status": "error", "error": "Bad request."}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Requested resource not found."}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"status": "error", "error": "An internal server error occurred."}), 500


if __name__ == "__main__":
    # Initialize database table and initial seed records
    database.init_db()
    # Ensure models are loaded on startup
    load_models()
    # Start the server on port 5000
    app.run(debug=True, host="0.0.0.0", port=5000)
