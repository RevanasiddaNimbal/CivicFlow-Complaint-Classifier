import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "civicflow.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database and migrates schema if needed."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT NOT NULL,
            cleaned_text TEXT NOT NULL,
            predicted_department TEXT NOT NULL,
            confidence REAL NOT NULL,
            urgency TEXT NOT NULL,
            urgency_reason TEXT,
            created_at TEXT NOT NULL,
            details_json TEXT
        )
    """)
    conn.commit()

    # Ensure details_json column exists if migrating an existing table
    cursor.execute("PRAGMA table_info(complaints_history)")
    columns = [row["name"] for row in cursor.fetchall()]
    if "details_json" not in columns:
        cursor.execute("ALTER TABLE complaints_history ADD COLUMN details_json TEXT")
        conn.commit()

    # Check if empty, seed with initial realistic records with complete details
    cursor.execute("SELECT COUNT(*) FROM complaints_history")
    count = cursor.fetchone()[0]
    if count == 0:
        initial_samples = [
            (
                "There are deep potholes and broken asphalt on the main avenue near the central station.",
                "there are deep potholes and broken asphalt on the main avenue near the central station",
                "Roads",
                0.935,
                "Medium Priority",
                "Disruption indicator detected: potholes",
                "2026-11-20 08:15:00",
                json.dumps({
                    "complaint": "There are deep potholes and broken asphalt on the main avenue near the central station.",
                    "clean_complaint": "there are deep potholes and broken asphalt on the main avenue near the central station",
                    "prediction": "Roads",
                    "confidence": 0.935,
                    "all_probabilities": {"Roads": 0.935, "Traffic": 0.030, "Water": 0.015, "Sanitation": 0.010, "Electricity": 0.010},
                    "explainability": {
                        "explanation": "The words related to 'potholes' and 'asphalt' contributed most strongly to the Roads department prediction.",
                        "contributing_words": [
                            {"word": "potholes", "weight": 1.95, "tfidf": 0.62, "contribution": 1.21, "impact_percentage": 62.5},
                            {"word": "asphalt", "weight": 1.45, "tfidf": 0.50, "contribution": 0.73, "impact_percentage": 37.5}
                        ],
                        "has_contributing_words": True
                    },
                    "urgency": {
                        "level": "Medium Priority",
                        "badge_class": "badge-priority-medium",
                        "matched_keywords": ["potholes"],
                        "reason": "Active disruption indicator detected: potholes",
                        "description": "Significant civic disruption requiring scheduled maintenance.",
                        "is_rule_based": True
                    },
                    "duplicate_detection": {
                        "is_similar": False,
                        "similarity_percentage": 0.0,
                        "message": "No similar complaints found."
                    },
                    "human_review": {
                        "needs_review": False,
                        "message": "Prediction confidence is sufficient for automatic routing.",
                        "detail": "Routing confidence is 93.5%, meeting the 60% verification standard."
                    },
                    "preprocessing": {
                        "original": "There are deep potholes and broken asphalt on the main avenue near the central station.",
                        "cleaned": "there are deep potholes and broken asphalt on the main avenue near the central station",
                        "steps": [
                            {"step_number": 1, "title": "Raw Input Text", "description": "Original citizen submission with original case and punctuation.", "output": "There are deep potholes and broken asphalt on the main avenue near the central station."},
                            {"step_number": 2, "title": "Lowercasing", "description": "Converted all letters to lowercase so capitalization does not affect results.", "output": "there are deep potholes and broken asphalt on the main avenue near the central station."},
                            {"step_number": 3, "title": "Punctuation Stripping", "description": "Removed commas, periods, and special symbols.", "output": "there are deep potholes and broken asphalt on the main avenue near the central station"},
                            {"step_number": 4, "title": "Whitespace Normalization", "description": "Cleaned up extra spaces and trimmed edges.", "output": "there are deep potholes and broken asphalt on the main avenue near the central station"},
                            {"step_number": 5, "title": "Civic Vocabulary Mapping", "description": "Identified civic keywords recognized by the model.", "output": "potholes, asphalt"}
                        ]
                    }
                })
            ),
            (
                "A drinking water supply pipeline burst near the city hospital and is causing heavy water wastage.",
                "a drinking water supply pipeline burst near the city hospital and is causing heavy water wastage",
                "Water",
                0.962,
                "High Priority",
                "Urgent safety hazard detected: burst",
                "2026-11-20 08:30:00",
                json.dumps({
                    "complaint": "A drinking water supply pipeline burst near the city hospital and is causing heavy water wastage.",
                    "clean_complaint": "a drinking water supply pipeline burst near the city hospital and is causing heavy water wastage",
                    "prediction": "Water",
                    "confidence": 0.962,
                    "all_probabilities": {"Water": 0.962, "Roads": 0.015, "Sanitation": 0.010, "Electricity": 0.008, "Traffic": 0.005},
                    "explainability": {
                        "explanation": "The words related to 'water', 'pipeline', and 'burst' contributed most strongly to the Water department prediction.",
                        "contributing_words": [
                            {"word": "water", "weight": 2.10, "tfidf": 0.65, "contribution": 1.36, "impact_percentage": 52.0},
                            {"word": "pipeline", "weight": 1.55, "tfidf": 0.52, "contribution": 0.81, "impact_percentage": 31.0},
                            {"word": "burst", "weight": 1.20, "tfidf": 0.37, "contribution": 0.44, "impact_percentage": 17.0}
                        ],
                        "has_contributing_words": True
                    },
                    "urgency": {
                        "level": "High Priority",
                        "badge_class": "badge-priority-high",
                        "matched_keywords": ["burst"],
                        "reason": "Urgent safety hazard detected: burst",
                        "description": "Immediate public safety or infrastructure risk detected. Prioritize rapid field response.",
                        "is_rule_based": True
                    },
                    "duplicate_detection": {
                        "is_similar": False,
                        "similarity_percentage": 0.0,
                        "message": "No similar complaints found."
                    },
                    "human_review": {
                        "needs_review": False,
                        "message": "Prediction confidence is sufficient for automatic routing.",
                        "detail": "Routing confidence is 96.2%, meeting the 60% verification standard."
                    },
                    "preprocessing": {
                        "original": "A drinking water supply pipeline burst near the city hospital and is causing heavy water wastage.",
                        "cleaned": "a drinking water supply pipeline burst near the city hospital and is causing heavy water wastage",
                        "steps": [
                            {"step_number": 1, "title": "Raw Input Text", "description": "Original citizen submission with original case and punctuation.", "output": "A drinking water supply pipeline burst near the city hospital and is causing heavy water wastage."},
                            {"step_number": 2, "title": "Lowercasing", "description": "Converted all letters to lowercase so capitalization does not affect results.", "output": "a drinking water supply pipeline burst near the city hospital and is causing heavy water wastage."},
                            {"step_number": 3, "title": "Punctuation Stripping", "description": "Removed commas, periods, and special symbols.", "output": "a drinking water supply pipeline burst near the city hospital and is causing heavy water wastage"},
                            {"step_number": 4, "title": "Whitespace Normalization", "description": "Cleaned up extra spaces and trimmed edges.", "output": "a drinking water supply pipeline burst near the city hospital and is causing heavy water wastage"},
                            {"step_number": 5, "title": "Civic Vocabulary Mapping", "description": "Identified civic keywords recognized by the model.", "output": "water, pipeline, burst"}
                        ]
                    }
                })
            ),
            (
                "High voltage electric wire is sparking dangerously on the transformer pole near school.",
                "high voltage electric wire is sparking dangerously on the transformer pole near school",
                "Electricity",
                0.978,
                "High Priority",
                "Urgent safety hazard detected: dangerous, sparking",
                "2026-11-20 08:45:00",
                json.dumps({
                    "complaint": "High voltage electric wire is sparking dangerously on the transformer pole near school.",
                    "clean_complaint": "high voltage electric wire is sparking dangerously on the transformer pole near school",
                    "prediction": "Electricity",
                    "confidence": 0.978,
                    "all_probabilities": {"Electricity": 0.978, "Roads": 0.008, "Traffic": 0.006, "Water": 0.004, "Sanitation": 0.004},
                    "explainability": {
                        "explanation": "The words related to 'electric', 'wire', 'transformer', and 'sparking' contributed most strongly to the Electricity department prediction.",
                        "contributing_words": [
                            {"word": "electric", "weight": 2.25, "tfidf": 0.58, "contribution": 1.30, "impact_percentage": 42.0},
                            {"word": "wire", "weight": 1.70, "tfidf": 0.48, "contribution": 0.82, "impact_percentage": 26.5},
                            {"word": "transformer", "weight": 1.50, "tfidf": 0.42, "contribution": 0.63, "impact_percentage": 20.3},
                            {"word": "sparking", "weight": 1.25, "tfidf": 0.28, "contribution": 0.35, "impact_percentage": 11.2}
                        ],
                        "has_contributing_words": True
                    },
                    "urgency": {
                        "level": "High Priority",
                        "badge_class": "badge-priority-high",
                        "matched_keywords": ["dangerous", "sparking"],
                        "reason": "Urgent safety hazard detected: dangerous, sparking",
                        "description": "Immediate public safety or infrastructure risk detected. Prioritize rapid field response.",
                        "is_rule_based": True
                    },
                    "duplicate_detection": {
                        "is_similar": False,
                        "similarity_percentage": 0.0,
                        "message": "No similar complaints found."
                    },
                    "human_review": {
                        "needs_review": False,
                        "message": "Prediction confidence is sufficient for automatic routing.",
                        "detail": "Routing confidence is 97.8%, meeting the 60% verification standard."
                    },
                    "preprocessing": {
                        "original": "High voltage electric wire is sparking dangerously on the transformer pole near school.",
                        "cleaned": "high voltage electric wire is sparking dangerously on the transformer pole near school",
                        "steps": [
                            {"step_number": 1, "title": "Raw Input Text", "description": "Original citizen submission with original case and punctuation.", "output": "High voltage electric wire is sparking dangerously on the transformer pole near school."},
                            {"step_number": 2, "title": "Lowercasing", "description": "Converted all letters to lowercase so capitalization does not affect results.", "output": "high voltage electric wire is sparking dangerously on the transformer pole near school."},
                            {"step_number": 3, "title": "Punctuation Stripping", "description": "Removed commas, periods, and special symbols.", "output": "high voltage electric wire is sparking dangerously on the transformer pole near school"},
                            {"step_number": 4, "title": "Whitespace Normalization", "description": "Cleaned up extra spaces and trimmed edges.", "output": "high voltage electric wire is sparking dangerously on the transformer pole near school"},
                            {"step_number": 5, "title": "Civic Vocabulary Mapping", "description": "Identified civic keywords recognized by the model.", "output": "voltage, electric, wire, sparking, transformer, pole"}
                        ]
                    }
                })
            ),
            (
                "Garbage dumpster bins are overflowing with rotting waste and foul smell across the street.",
                "garbage dumpster bins are overflowing with rotting waste and foul smell across the street",
                "Sanitation",
                0.941,
                "Medium Priority",
                "Disruption indicator detected: overflowing, stink, odor",
                "2026-11-20 09:00:00",
                json.dumps({
                    "complaint": "Garbage dumpster bins are overflowing with rotting waste and foul smell across the street.",
                    "clean_complaint": "garbage dumpster bins are overflowing with rotting waste and foul smell across the street",
                    "prediction": "Sanitation",
                    "confidence": 0.941,
                    "all_probabilities": {"Sanitation": 0.941, "Water": 0.025, "Roads": 0.015, "Electricity": 0.010, "Traffic": 0.009},
                    "explainability": {
                        "explanation": "The words related to 'garbage', 'dumpster', and 'waste' contributed most strongly to the Sanitation department prediction.",
                        "contributing_words": [
                            {"word": "garbage", "weight": 2.15, "tfidf": 0.60, "contribution": 1.29, "impact_percentage": 51.2},
                            {"word": "dumpster", "weight": 1.60, "tfidf": 0.45, "contribution": 0.72, "impact_percentage": 28.6},
                            {"word": "waste", "weight": 1.35, "tfidf": 0.38, "contribution": 0.51, "impact_percentage": 20.2}
                        ],
                        "has_contributing_words": True
                    },
                    "urgency": {
                        "level": "Medium Priority",
                        "badge_class": "badge-priority-medium",
                        "matched_keywords": ["overflowing", "smell"],
                        "reason": "Active disruption indicator detected: overflowing, smell",
                        "description": "Significant civic disruption requiring scheduled maintenance.",
                        "is_rule_based": True
                    },
                    "duplicate_detection": {
                        "is_similar": False,
                        "similarity_percentage": 0.0,
                        "message": "No similar complaints found."
                    },
                    "human_review": {
                        "needs_review": False,
                        "message": "Prediction confidence is sufficient for automatic routing.",
                        "detail": "Routing confidence is 94.1%, meeting the 60% verification standard."
                    },
                    "preprocessing": {
                        "original": "Garbage dumpster bins are overflowing with rotting waste and foul smell across the street.",
                        "cleaned": "garbage dumpster bins are overflowing with rotting waste and foul smell across the street",
                        "steps": [
                            {"step_number": 1, "title": "Raw Input Text", "description": "Original citizen submission with original case and punctuation.", "output": "Garbage dumpster bins are overflowing with rotting waste and foul smell across the street."},
                            {"step_number": 2, "title": "Lowercasing", "description": "Converted all letters to lowercase so capitalization does not affect results.", "output": "garbage dumpster bins are overflowing with rotting waste and foul smell across the street."},
                            {"step_number": 3, "title": "Punctuation Stripping", "description": "Removed commas, periods, and special symbols.", "output": "garbage dumpster bins are overflowing with rotting waste and foul smell across the street"},
                            {"step_number": 4, "title": "Whitespace Normalization", "description": "Cleaned up extra spaces and trimmed edges.", "output": "garbage dumpster bins are overflowing with rotting waste and foul smell across the street"},
                            {"step_number": 5, "title": "Civic Vocabulary Mapping", "description": "Identified civic keywords recognized by the model.", "output": "garbage, dumpster, bins, overflowing, waste, smell"}
                        ]
                    }
                })
            ),
            (
                "Severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking.",
                "severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking",
                "Traffic",
                0.954,
                "High Priority",
                "Urgent safety hazard detected: severe",
                "2026-11-20 09:15:00",
                json.dumps({
                    "complaint": "Severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking.",
                    "clean_complaint": "severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking",
                    "prediction": "Traffic",
                    "confidence": 0.954,
                    "all_probabilities": {"Traffic": 0.954, "Roads": 0.020, "Sanitation": 0.010, "Electricity": 0.008, "Water": 0.008},
                    "explainability": {
                        "explanation": "The words related to 'traffic', 'congestion', and 'gridlock' contributed most strongly to the Traffic department prediction.",
                        "contributing_words": [
                            {"word": "traffic", "weight": 2.20, "tfidf": 0.62, "contribution": 1.36, "impact_percentage": 50.8},
                            {"word": "congestion", "weight": 1.65, "tfidf": 0.46, "contribution": 0.76, "impact_percentage": 28.4},
                            {"word": "gridlock", "weight": 1.40, "tfidf": 0.40, "contribution": 0.56, "impact_percentage": 20.8}
                        ],
                        "has_contributing_words": True
                    },
                    "urgency": {
                        "level": "High Priority",
                        "badge_class": "badge-priority-high",
                        "matched_keywords": ["severe"],
                        "reason": "Urgent safety hazard detected: severe",
                        "description": "Immediate public safety or infrastructure risk detected. Prioritize rapid field response.",
                        "is_rule_based": True
                    },
                    "duplicate_detection": {
                        "is_similar": False,
                        "similarity_percentage": 0.0,
                        "message": "No similar complaints found."
                    },
                    "human_review": {
                        "needs_review": False,
                        "message": "Prediction confidence is sufficient for automatic routing.",
                        "detail": "Routing confidence is 95.4%, meeting the 60% verification standard."
                    },
                    "preprocessing": {
                        "original": "Severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking.",
                        "cleaned": "severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking",
                        "steps": [
                            {"step_number": 1, "title": "Raw Input Text", "description": "Original citizen submission with original case and punctuation.", "output": "Severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking."},
                            {"step_number": 2, "title": "Lowercasing", "description": "Converted all letters to lowercase so capitalization does not affect results.", "output": "severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking."},
                            {"step_number": 3, "title": "Punctuation Stripping", "description": "Removed commas, periods, and special symbols.", "output": "severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking"},
                            {"step_number": 4, "title": "Whitespace Normalization", "description": "Cleaned up extra spaces and trimmed edges.", "output": "severe traffic congestion and gridlock at the metro signal due to illegal vehicle parking"},
                            {"step_number": 5, "title": "Civic Vocabulary Mapping", "description": "Identified civic keywords recognized by the model.", "output": "traffic, congestion, gridlock, signal, vehicle, parking"}
                        ]
                    }
                })
            )
        ]
        cursor.executemany("""
            INSERT INTO complaints_history 
            (original_text, cleaned_text, predicted_department, confidence, urgency, urgency_reason, created_at, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, initial_samples)
        conn.commit()
    conn.close()

def insert_prediction(original_text, cleaned_text, predicted_dept, confidence, urgency, urgency_reason, details=None):
    """Inserts a newly classified complaint record into the history table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    details_str = json.dumps(details) if details is not None else None
    cursor.execute("""
        INSERT INTO complaints_history 
        (original_text, cleaned_text, predicted_department, confidence, urgency, urgency_reason, created_at, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (original_text, cleaned_text, predicted_dept, float(confidence), urgency, urgency_reason, now_str, details_str))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return new_id

def get_prediction_by_id(record_id):
    """Retrieves a single complaint prediction record by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, original_text, cleaned_text, predicted_department, confidence, urgency, urgency_reason, created_at, details_json
        FROM complaints_history
        WHERE id = ?
    """, (record_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_all_complaints_for_similarity():
    """Retrieves all previous complaint texts for duplicate detection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, original_text, cleaned_text, predicted_department, created_at 
        FROM complaints_history 
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_history(limit=50):
    """Retrieves recent complaint predictions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, original_text, cleaned_text, predicted_department, confidence, urgency, urgency_reason, created_at 
        FROM complaints_history 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def clear_history():
    """Clears all records from the complaint history table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM complaints_history")
    conn.commit()
    conn.close()
