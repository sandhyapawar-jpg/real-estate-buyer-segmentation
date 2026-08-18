import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)

# Load final segmented dataset
DATA_FILE = "Real Estate Buyer Segmentation Final.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

st.title("🏠 Real Estate Buyer Segmentation")
st.write(
    "Machine Learning based Buyer Segmentation and Investment Profiling "
    "for Real Estate Market Intelligence"
)

st.divider()

# Basic information
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Buyers", len(df))

with col2:
    st.metric("Number of Segments", df["segment"].nunique())

with col3:
    st.metric("Average Satisfaction", round(df["satisfaction_score"].mean(), 2))

st.subheader("📊 Buyer Distribution by Segment")

segment_counts = df["segment"].value_counts().sort_index()
st.bar_chart(segment_counts)

st.divider()

# Segment selection
st.subheader("🔎 Explore a Buyer Segment")

selected_segment = st.selectbox(
    "Select a segment:",
    sorted(df["segment"].unique())
)

segment_data = df[df["segment"] == selected_segment]

st.write(f"### Segment {selected_segment}")
st.write(f"Number of buyers: **{len(segment_data)}**")

col1, col2 = st.columns(2)

with col1:
    st.write("**Average Age**")
    st.write(round(segment_data["age"].mean(), 2))

    st.write("**Average Satisfaction**")
    st.write(round(segment_data["satisfaction_score"].mean(), 2))

with col2:
    st.write("**Main Country**")
    st.write(segment_data["country"].mode()[0])

    st.write("**Main Acquisition Purpose**")
    st.write(segment_data["acquisition_purpose"].mode()[0])

st.subheader("Buyer Characteristics")

profile = pd.DataFrame({
    "Characteristic": [
        "Client Type",
        "Gender",
        "Country",
        "Purpose",
        "Loan Applied",
        "Referral Channel"
    ],
    "Most Common Value": [
        segment_data["client_type"].mode()[0],
        segment_data["gender"].mode()[0],
        segment_data["country"].mode()[0],
        segment_data["acquisition_purpose"].mode()[0],
        segment_data["loan_applied"].mode()[0],
        segment_data["referral_channel"].mode()[0]
    ]
})

st.table(profile)

st.divider()

st.subheader("📋 Buyer Data")

st.dataframe(segment_data, use_container_width=True)

st.success(
    "This dashboard provides an interactive view of the buyer segments "
    "created using K-Means clustering."
)
