import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.jsonl")

if not os.path.exists(FEEDBACK_FILE):
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(FEEDBACK_FILE, "w") as f:
        pass  

def save_user_feedback(query, answer, rating, comment):
    feedback_entry = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "query": query,
        "model_answer": answer,
        "rating": rating,  # 👍 or 👎
        "comment": comment
    }

    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
