import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Forex Lead Intelligence", layout="wide")

st.title("Forex Lead Intelligence Dashboard")

st.write(
"AI system that ranks leads based on probability of becoming a forex trading customer."
)

# Load model
model = joblib.load("../models/lead_scoring_model.pkl")

# Load dataset
df = pd.read_csv("../data/leads.csv")

# Prepare features
features = df.drop(columns=["deposit", "lead_id"])

# Predict scores
scores = model.predict_proba(features)[:,1]

df["lead_score"] = scores


# Assign priority
def assign_priority(score):

    if score > 0.7:
        return "HIGH"

    elif score > 0.4:
        return "MEDIUM"

    else:
        return "LOW"


df["priority"] = df["lead_score"].apply(assign_priority)


# AI reasoning
def explain_lead(row):

    reasons = []

    if row["demo_requested"] == 1:
        reasons.append("Demo requested")

    if row["asked_about_leverage"] == 1:
        reasons.append("Asked about leverage")

    if row["answered_call"] == 1:
        reasons.append("Answered call")

    if row["lead_source"] == "webinar":
        reasons.append("Webinar lead")

    if len(reasons) == 0:
        reasons.append("Low engagement")

    return ", ".join(reasons)


df["signals"] = df.apply(explain_lead, axis=1)

# Sort leads
df = df.sort_values("lead_score", ascending=False)


# -----------------------
# Dashboard Metrics
# -----------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Leads", len(df))
col2.metric("High Priority Leads", len(df[df.priority=="HIGH"]))
col3.metric("Medium Priority Leads", len(df[df.priority=="MEDIUM"]))


# -----------------------
# Filters
# -----------------------
st.sidebar.header("Filter Leads")

priority_filter = st.sidebar.multiselect(
    "Priority",
    options=df["priority"].unique(),
    default=df["priority"].unique()
)

state_filter = st.sidebar.multiselect(
    "State",
    options=df["state"].unique(),
    default=df["state"].unique()
)

profession_filter = st.sidebar.multiselect(
    "Profession",
    options=df["profession"].unique(),
    default=df["profession"].unique()
)

source_filter = st.sidebar.multiselect(
    "Lead Source",
    options=df["lead_source"].unique(),
    default=df["lead_source"].unique()
)

score_filter = st.sidebar.slider(
    "Minimum Lead Score",
    0.0,
    1.0,
    0.0
)


# Apply filters
filtered_df = df[
    (df["priority"].isin(priority_filter)) &
    (df["state"].isin(state_filter)) &
    (df["profession"].isin(profession_filter)) &
    (df["lead_source"].isin(source_filter)) &
    (df["lead_score"] >= score_filter)
]


# -----------------------
# Lead Table
# -----------------------
st.write("### Leads for Sales Outreach")

st.dataframe(
    filtered_df[
        [
            "lead_id",
            "state",
            "profession",
            "lead_source",
            "lead_score",
            "priority",
            "signals"
        ]
    ],
    use_container_width=True
)


# -----------------------
# Score Distribution
# -----------------------
st.write("### Lead Score Distribution")

st.bar_chart(df["lead_score"])