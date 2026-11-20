import unittest
import json
import os
import database
from app import app, load_models, clean_text, detect_urgency, explain_prediction, check_duplicate_complaint, check_human_review, get_preprocessing_preview

class CivicFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        database.init_db()
        loaded = load_models()
        assert loaded, "Models must load successfully"
        cls.client = app.test_client()

    def test_database_operations(self):
        print("\n--- Testing Database Operations ---")
        history = database.get_history(10)
        self.assertGreaterEqual(len(history), 5, "Database should have initial seeded records")
        print(f"Initial history count: {len(history)}")

        # Insert a test prediction
        new_id = database.insert_prediction(
            original_text="Unit test complaint",
            cleaned_text="unit test complaint",
            predicted_dept="Roads",
            confidence=0.91,
            urgency="Low Priority",
            urgency_reason="Unit test"
        )
        self.assertIsNotNone(new_id)
        print(f"Inserted record ID: {new_id}")

    def test_clean_text_and_preprocessing(self):
        print("\n--- Testing Text Preprocessing Preview ---")
        raw = "   Deep POTHOLES on Main St., near #44!!   "
        preview = get_preprocessing_preview(raw)
        self.assertEqual(preview["cleaned"], "deep potholes on main st near 44")
        self.assertEqual(len(preview["steps"]), 5)
        print("Preprocessing steps output correctly verified:", preview["cleaned"])

    def test_urgency_detection(self):
        print("\n--- Testing Urgency Detection ---")
        # High Priority
        high_res = detect_urgency("Emergency! A massive water pipe burst and is flooding the road!")
        self.assertEqual(high_res["level"], "High Priority")
        print("High priority detected:", high_res["reason"])

        # Medium Priority
        med_res = detect_urgency("There is a large pothole and traffic delay on the crossroad.")
        self.assertEqual(med_res["level"], "Medium Priority")
        print("Medium priority detected:", med_res["reason"])

        # Low Priority
        low_res = detect_urgency("Please repaint the street sign when convenient.")
        self.assertEqual(low_res["level"], "Low Priority")
        print("Low priority detected:", low_res["reason"])

    def test_human_review_alert(self):
        print("\n--- Testing Human Review Alert ---")
        high_conf = check_human_review(0.92, threshold=0.60)
        self.assertFalse(high_conf["needs_review"])
        self.assertEqual(high_conf["alert_type"], "success")

        low_conf = check_human_review(0.45, threshold=0.60)
        self.assertTrue(low_conf["needs_review"])
        self.assertEqual(low_conf["alert_type"], "warning")
        print("Human review alert threshold verified.")

    def test_all_five_categories_prediction(self):
        print("\n--- Testing All 5 Complaint Categories Prediction ---")
        # Standard high-confidence civic complaints for all 5 departments
        categories_samples = [
            ("There are deep potholes and broken asphalt on the main avenue.", "Roads"),
            ("A water supply pipe burst near the park and is wasting clean drinking water.", "Water"),
            ("The power transformer is sparking dangerously and causing a total blackout.", "Electricity"),
            ("Garbage bins are overflowing and no one has collected waste in a week.", "Sanitation"),
            ("A heavy truck is double parked at the intersection, causing a massive gridlock.", "Traffic")
        ]

        for sample_text, expected_dept in categories_samples:
            response = self.client.post("/predict", json={"complaint": sample_text})
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["prediction"], expected_dept)
            self.assertGreater(data["confidence"], 0.70)
            explain = data["explainability"]
            self.assertTrue(explain["has_contributing_words"])
            self.assertGreater(len(explain["contributing_words"]), 0)
            print(f"Category: {expected_dept} -> Predicted: {data['prediction']} ({data['confidence']*100:.1f}%) | Top contributing word: {explain['contributing_words'][0]['word']} (impact: {explain['contributing_words'][0]['impact_percentage']}%)")

    def test_low_confidence_review_trigger(self):
        print("\n--- Testing Low-Confidence Human Review Alert ---")
        # Ambiguous complaint that does not contain strong civic keywords
        ambiguous_text = "The general public area seems somewhat dark and needs general attention."
        response = self.client.post("/predict", json={"complaint": ambiguous_text})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        rev = data["human_review"]
        self.assertTrue(rev["needs_review"])
        self.assertEqual(rev["alert_type"], "warning")
        print(f"Low confidence trigger confirmed: {data['confidence']*100:.1f}% -> {rev['message']}")

    def test_duplicate_detection(self):
        print("\n--- Testing Duplicate Detection ---")
        # 1. Exact/highly similar complaint to an existing seeded record
        response = self.client.post("/predict", json={
            "complaint": "There are deep potholes and broken asphalt on the main avenue near central station."
        })
        data = json.loads(response.data)
        dup = data["duplicate_detection"]
        self.assertTrue(dup["is_similar"], "Should detect duplicate similarity >= 60%")
        self.assertGreaterEqual(dup["similarity_percentage"], 60.0)
        print(f"Similar complaint detected! Similarity: {dup['similarity_percentage']}% | Matched: {dup['similar_complaint'][:40]}...")

        # 2. Distinct, never-before-seen complaint with random unique tokens
        import uuid
        u = uuid.uuid4().hex[:6]
        unique_text = f"AlphaBeta{u} GammaDelta{u} EpsilonZeta{u} ThetaIota{u} KappaLambda{u}"
        response_diff = self.client.post("/predict", json={"complaint": unique_text})
        data_diff = json.loads(response_diff.data)
        dup_diff = data_diff["duplicate_detection"]
        self.assertFalse(dup_diff["is_similar"])
        print(f"Unique complaint correctly flagged as non-duplicate. Similarity: {dup_diff['similarity_percentage']}%")

    def test_empty_input_validation(self):
        print("\n--- Testing Empty Input Validation ---")
        res = self.client.post("/predict", json={"complaint": "   "})
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "error")
        print("Empty complaint properly rejected with 400 status code.")

    def test_api_history_and_clear(self):
        print("\n--- Testing History API ---")
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertGreater(data["count"], 0)
        print(f"History API returned {data['count']} items.")

        # Test single record details API (without running new prediction)
        first_id = data["history"][0]["id"]
        res_detail = self.client.get(f"/api/history/{first_id}")
        self.assertEqual(res_detail.status_code, 200)
        detail_data = json.loads(res_detail.data)
        self.assertEqual(detail_data["status"], "success")
        self.assertIn("details", detail_data)
        self.assertIn("prediction", detail_data["details"])
        print(f"History detail API verified for record #{first_id}: Department = {detail_data['details']['prediction']}")

    def test_home_page_rendering(self):
        print("\n--- Testing Home Page Template Rendering ---")
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"CivicFlow", res.data)
        self.assertIn(b"Submit a Complaint", res.data)
        self.assertIn(b"AI Insights", res.data)
        self.assertIn(b"Complaint Analysis", res.data)
        self.assertIn(b"Complaint History", res.data)
        print("Frontend template rendered successfully with 200 OK.")

if __name__ == "__main__":
    unittest.main()
