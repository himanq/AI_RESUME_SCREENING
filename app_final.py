from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai_resume_screening.csv"
MODEL_PATH = BASE_DIR / "resume_screening_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"

TARGET = "shortlisted"

FEATURES = [
    "years_experience",
    "skills_match_score",
    "education_level",
    "project_count",
    "resume_length",
    "github_activity",
]


st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Trained model not found. Run `python train_model.py` "
            "before starting the app."
        )
        st.stop()

    # Load only a model artifact that you created/trust.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text())


df = load_data()
pipeline = load_model()
metrics = load_metrics()

missing = [column for column in FEATURES + [TARGET] if column not in df.columns]
if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()


st.title("📄 AI Resume Screening System")
st.caption(
    "Machine-learning based candidate screening using structured resume features."
)


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Candidate Screening",
        "Model Performance",
        "Feature Importance",
    ],
)


if page == "Dashboard":
    st.header("📊 Dashboard")

    total_candidates = len(df)
    shortlisted = int((df[TARGET] == "Yes").sum())
    not_shortlisted = int((df[TARGET] == "No").sum())
    shortlist_rate = (shortlisted / total_candidates * 100) if total_candidates else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Candidates", f"{total_candidates:,}")
    col2.metric("Shortlisted", f"{shortlisted:,}")
    col3.metric("Not Shortlisted", f"{not_shortlisted:,}")
    col4.metric("Shortlisting Rate", f"{shortlist_rate:.2f}%")

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Candidate Shortlisting Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set_xlabel("Shortlisted")
    ax.set_ylabel("Number of Candidates")
    ax.set_title("Shortlisting Distribution")
    st.pyplot(fig)
    plt.close(fig)


elif page == "Candidate Screening":
    st.header("🤖 Candidate Screening")
    st.write("Enter structured candidate information to generate a screening prediction.")

    col1, col2 = st.columns(2)

    with col1:
        experience = st.number_input(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=2,
            step=1,
        )

        skills = st.slider(
            "Skills Match Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.1,
        )

        education_options = sorted(df["education_level"].dropna().unique())
        education = st.selectbox("Education Level", education_options)

    with col2:
        projects = st.number_input(
            "Project Count",
            min_value=0,
            max_value=100,
            value=3,
            step=1,
        )

        resume_length = st.number_input(
            "Resume Length",
            min_value=1,
            max_value=5000,
            value=500,
            step=1,
        )

        github = st.number_input(
            "GitHub Activity",
            min_value=0,
            max_value=5000,
            value=100,
            step=1,
        )

    st.divider()

    if st.button("🔍 Screen Candidate", type="primary"):
        candidate = pd.DataFrame(
            [{
                "years_experience": experience,
                "skills_match_score": skills,
                "education_level": education,
                "project_count": projects,
                "resume_length": resume_length,
                "github_activity": github,
            }]
        )

        prediction = int(pipeline.predict(candidate)[0])
        probability = float(pipeline.predict_proba(candidate)[0, 1])

        st.subheader("Screening Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.success("✅ Candidate Predicted: SHORTLISTED")
            else:
                st.warning("⚠️ Candidate Predicted: NOT SHORTLISTED")

        with result_col2:
            st.metric(
                "Model Shortlisting Probability",
                f"{probability * 100:.2f}%",
            )

        st.caption(
            "This is a model prediction, not a hiring decision. "
            "Review candidates manually before making employment decisions."
        )


elif page == "Model Performance":
    st.header("📈 Model Performance")

    if metrics is None:
        st.warning("Metrics file not found.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", "Random Forest")
        col2.metric("Test Samples", f"{metrics['test_rows']:,}")
        col3.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        col4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        st.divider()

        st.subheader("Classification Report")
        report_df = pd.DataFrame(metrics["classification_report"]).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=["Not Shortlisted", "Shortlisted"],
            yticklabels=["Not Shortlisted", "Shortlisted"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)


elif page == "Feature Importance":
    st.header("🔍 Feature Importance")

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False)

    st.dataframe(importance_df.round(4), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        ax=ax,
    )
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)
    plt.close(fig)
