from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix


# Load trained TF-IDF vectorizer
tfidf = joblib.load("model/tfidf.pkl")

# Load trained Logistic Regression model
model = joblib.load("model/tfidf_model.pkl")


# Create FastAPI application
app = FastAPI(
    title="Quora Duplicate Question Detection API",
    description="Predict whether two questions are duplicates.",
    version="1.0.0"
)


# Input format
class QuestionPair(BaseModel):
    question1: str
    question2: str


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Quora Duplicate Question Detection API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: QuestionPair):

    # Get questions
    q1 = [data.question1]
    q2 = [data.question2]

    # Convert questions to TF-IDF vectors
    q1_tfidf = tfidf.transform(q1)
    q2_tfidf = tfidf.transform(q2)

    # 1. Absolute difference
    difference = abs(q1_tfidf - q2_tfidf)

    # 2. Element-wise multiplication
    product = q1_tfidf.multiply(q2_tfidf)

    # 3. Cosine similarity
    cosine = cosine_similarity(
        q1_tfidf,
        q2_tfidf
    )[0][0]

    cosine_feature = csr_matrix([[cosine]])

    # Combine the same features used during training
    final_features = hstack([
        difference,
        product,
        cosine_feature
    ])

    # Make prediction
    prediction = model.predict(final_features)[0]

    # Prediction probability
    probability = model.predict_proba(final_features)[0][1]

    return {
        "question1": data.question1,
        "question2": data.question2,
        "duplicate": int(prediction),
        "probability": round(float(probability), 4)
    }