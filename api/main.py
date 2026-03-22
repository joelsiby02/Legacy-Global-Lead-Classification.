import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("../models/lead_scoring_model.pkl")


# -----------------------------
# Initialize API
# -----------------------------
app = FastAPI(
    title="Forex Lead Intelligence API",
    description="AI system that predicts probability of lead converting to forex customer"
)


# -----------------------------
# Input Schema
# -----------------------------
class LeadData(BaseModel):
    state: str
    city_tier: int
    age: int
    profession: str
    lead_source: str
    answered_call: int
    asked_about_leverage: int
    demo_requested: int


# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def home():
    return {"message": "Lead Intelligence API running"}


# -----------------------------
# Lead Scoring Endpoint
# -----------------------------
@app.post("/score")
def score_lead(data: LeadData):

    # Convert input to dataframe
    df = pd.DataFrame([data.dict()])

    # Predict probability
    score = model.predict_proba(df)[0][1]

    # Determine priority
    if score > 0.7:
        priority = "HIGH"
    elif score > 0.4:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # Generate reasoning signals
    reasons = []

    if data.demo_requested == 1:
        reasons.append("Lead requested demo trading account")

    if data.asked_about_leverage == 1:
        reasons.append("Lead asked about leverage")

    if data.answered_call == 1:
        reasons.append("Lead answered sales call")

    if data.lead_source == "webinar":
        reasons.append("Lead joined trading webinar")

    if data.profession in ["Business", "Trader"]:
        reasons.append("Profession associated with trading interest")

    if len(reasons) == 0:
        reasons.append("Lead shows limited engagement signals")

    # Return response
    return {
        "lead_score": round(float(score), 2),
        "priority": priority,
        "reasons": reasons
    }