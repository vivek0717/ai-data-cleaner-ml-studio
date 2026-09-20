import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
from io import BytesIO
from google import genai

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(
    page_title="AI Data Cleaner & ML Studio",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
:root {
    --bg: #06111f;
    --bg2: #0b1728;
    --card: rgba(10, 20, 34, 0.92);
    --card2: rgba(15, 28, 46, 0.95);
    --border: rgba(100, 170, 255, 0.16);
    --text: #eaf2ff;
    --muted: #9db2ce;
    --accent: #38bdf8;
    --accent2: #14b8a6;
}

html, body, [class*="css"] {
    color: var(--text);
}

.stApp {
    background:
        radial-gradient(circle at top right, rgba(56,189,248,0.10), transparent 22%),
        radial-gradient(circle at top left, rgba(20,184,166,0.08), transparent 20%),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    color: var(--text);
}

.block-container {
    max-width: 1280px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

.hero-wrap {
    background: linear-gradient(135deg, rgba(15, 30, 50, 0.98), rgba(8, 18, 32, 0.98));
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 28px;
    margin-bottom: 18px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.28);
}

.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.1;
    color: white;
    margin-bottom: 0.35rem;
}

.hero-sub {
    color: var(--muted);
    font-size: 1.02rem;
    max-width: 900px;
}

.soft-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 18px;
    margin: 12px 0;
    box-shadow: 0 10px 24px rgba(0,0,0,0.16);
}

.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(56,189,248,0.12);
    color: #bfeaff;
    border: 1px solid rgba(56,189,248,0.18);
    font-size: 0.85rem;
    margin-bottom: 10px;
}

div[data-testid="stMetric"] {
    background: var(--card2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 12px 14px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea,
.stTextInput input,
.stNumberInput input {
    background: rgba(12, 22, 36, 0.95) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(100,170,255,0.14) !important;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #0ea5e9, #14b8a6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.58rem 1rem;
    font-weight: 700;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    filter: brightness(1.07);
    transform: translateY(-1px);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(12, 22, 36, 0.92);
    border: 1px solid rgba(100,170,255,0.12);
    border-radius: 12px 12px 0 0;
    padding: 10px 18px;
}

hr {
    border: none;
    height: 1px;
    background: rgba(100,170,255,0.12);
    margin: 18px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="badge">Dark interactive workspace</div>
    <div class="hero-title">AI Data Cleaner & ML Studio</div>
    <div class="hero-sub">
        Clean your dataset with AI and manual controls, explore charts, then either test one model deeply or compare multiple models side by side.
    </div>
</div>
""", unsafe_allow_html=True)


def normalize_text(text):
    return str(text).strip().lower().replace(" ", "_")


def load_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        file_type = "csv"
    elif file_name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(uploaded_file)
        file_type = "excel"
    else:
        raise ValueError("Unsupported file type. Please upload CSV or Excel.")
    return df, file_type


def remove_exact_duplicates(df):
    before_rows = len(df)
    cleaned_df = df.drop_duplicates().copy()
    removed_rows = before_rows - len(cleaned_df)
    return cleaned_df, f"Removed {removed_rows} exact duplicate rows."


def trim_text_columns(df):
    cleaned_df = df.copy()
    text_columns = cleaned_df.select_dtypes(include="object").columns
    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return cleaned_df, "Trimmed leading and trailing spaces in text columns."


def standardize_text_case(df):
    cleaned_df = df.copy()
    text_columns = cleaned_df.select_dtypes(include="object").columns
    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].apply(lambda x: x.title() if isinstance(x, str) else x)
    return cleaned_df, "Standardized text case in text columns."


def standardize_column_names(df):
    cleaned_df = df.copy()
    cleaned_df.columns = (
        cleaned_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return cleaned_df, "Standardized column names."


def find_matching_column(df, requested_name):
    normalized_requested = normalize_text(requested_name)
    column_map = {normalize_text(col): col for col in df.columns}
    return column_map.get(normalized_requested)


def drop_column_by_name(df, column_name):
    cleaned_df = df.copy()
    matched_column = find_matching_column(cleaned_df, column_name)
    if matched_column:
        cleaned_df = cleaned_df.drop(columns=[matched_column])
        return cleaned_df, f"Dropped column '{matched_column}'."
    return cleaned_df, f"Column '{column_name}' was not found."


def fill_missing_text_with_unknown(df):
    cleaned_df = df.copy()
    text_columns = cleaned_df.select_dtypes(include="object").columns
    if len(text_columns) > 0:
        cleaned_df[text_columns] = cleaned_df[text_columns].fillna("Unknown")
    return cleaned_df, "Filled missing values in text columns with 'Unknown'."


def fill_missing_numeric_with_median(df):
    cleaned_df = df.copy()
    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns
    for col in numeric_columns:
        if cleaned_df[col].isna().sum() > 0:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    return cleaned_df, "Filled missing numeric values with median."


def drop_rows_with_many_missing(df, threshold_percent=50):
    cleaned_df = df.copy()
    minimum_non_missing = max(1, int(len(cleaned_df.columns) * (1 - threshold_percent / 100)))
    before_rows = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(thresh=minimum_non_missing)
    removed_rows = before_rows - len(cleaned_df)
    return cleaned_df, f"Removed {removed_rows} rows with too many missing values."


def make_downloadable_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def make_downloadable_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned_data")
    return output.getvalue()


def generate_warnings_and_notes(original_df, cleaned_df, actions_applied):
    warnings = []
    removed_rows = len(original_df) - len(cleaned_df)
    removed_columns = len(original_df.columns) - len(cleaned_df.columns)

    if removed_rows > 0:
        warnings.append(f"{removed_rows} rows were removed during cleaning.")
    if removed_columns > 0:
        warnings.append(f"{removed_columns} columns were removed during cleaning.")
    if any("Dropped column" in action for action in actions_applied):
        warnings.append("A column drop action was applied. Please confirm no important business field was removed.")
    if any("median" in action.lower() for action in actions_applied):
        warnings.append("Numeric missing values were filled with the median, which may change analytical results.")
    if any("Unknown" in action for action in actions_applied):
        warnings.append("Missing text values were replaced with 'Unknown'.")
    if not warnings:
        warnings.append("No major warnings. Review the cleaned file before final use.")
    return warnings


def generate_html_report(original_df, cleaned_df, actions_applied):
    original_rows = len(original_df)
    cleaned_rows = len(cleaned_df)
    original_columns = len(original_df.columns)
    cleaned_columns = len(cleaned_df.columns)
    original_missing = int(original_df.isna().sum().sum())
    cleaned_missing = int(cleaned_df.isna().sum().sum())
    original_duplicates = int(original_df.duplicated().sum())
    cleaned_duplicates = int(cleaned_df.duplicated().sum())

    actions_html = "".join([f"<li>{action}</li>" for action in actions_applied]) or "<li>No actions applied</li>"
    warnings = generate_warnings_and_notes(original_df, cleaned_df, actions_applied)
    warnings_html = "".join([f"<li>{warning}</li>" for warning in warnings])

    html = f"""
    <html>
    <head>
        <title>Data Cleaning Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f9f9f9;
                color: #222;
            }}
            h1, h2 {{ color: #0f4c5c; }}
            .box {{
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: left;
            }}
            th {{ background-color: #eaf4f4; }}
        </style>
    </head>
    <body>
        <h1>Data Cleaning Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <div class="box">
            <h2>Dataset Summary</h2>
            <table>
                <tr><th>Metric</th><th>Before Cleaning</th><th>After Cleaning</th></tr>
                <tr><td>Rows</td><td>{original_rows}</td><td>{cleaned_rows}</td></tr>
                <tr><td>Columns</td><td>{original_columns}</td><td>{cleaned_columns}</td></tr>
                <tr><td>Missing Values</td><td>{original_missing}</td><td>{cleaned_missing}</td></tr>
                <tr><td>Duplicate Rows</td><td>{original_duplicates}</td><td>{cleaned_duplicates}</td></tr>
            </table>
        </div>
        <div class="box">
            <h2>Actions Performed</h2>
            <ul>{actions_html}</ul>
        </div>
        <div class="box">
            <h2>Warnings / Notes</h2>
            <ul>{warnings_html}</ul>
        </div>
    </body>
    </html>
    """
    return html.encode("utf-8")


def clean_json_text(text_output):
    text_output = text_output.strip()
    if text_output.startswith("```json"):
        text_output = text_output.replace("```json", "", 1).strip()
    elif text_output.startswith("```"):
        text_output = text_output.replace("```", "", 1).strip()
    if text_output.endswith("```"):
        text_output = text_output[:-3].strip()
    return text_output


def get_api_plan(user_request, columns, api_key):
    client = genai.Client(api_key=api_key)

    allowed_actions = [
        "remove_duplicates",
        "trim_spaces",
        "standardize_text_case",
        "standardize_columns",
        "drop_column",
        "fill_text_missing",
        "fill_numeric_missing",
        "drop_rows_many_missing"
    ]

    prompt = f"""
You are a data cleaning planner.
Return only valid JSON.
Do not explain anything.
Do not use markdown.
Choose only from these allowed actions:
{allowed_actions}

Available columns:
{columns}

User request:
{user_request}

Return JSON exactly like this:
{{
  "actions": [
    {{"action": "drop_column", "column_name": "latitude"}},
    {{"action": "trim_spaces"}}
  ]
}}
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    raw_text = response.text
    cleaned_text = clean_json_text(raw_text)
    parsed_plan = json.loads(cleaned_text)
    return parsed_plan, raw_text


def run_actions_from_plan(df, plan):
    working_df = df.copy()
    action_messages = []

    for item in plan.get("actions", []):
        action = item.get("action")

        if action == "remove_duplicates":
            working_df, msg = remove_exact_duplicates(working_df)
        elif action == "trim_spaces":
            working_df, msg = trim_text_columns(working_df)
        elif action == "standardize_text_case":
            working_df, msg = standardize_text_case(working_df)
        elif action == "standardize_columns":
            working_df, msg = standardize_column_names(working_df)
        elif action == "drop_column":
            working_df, msg = drop_column_by_name(working_df, item.get("column_name", ""))
        elif action == "fill_text_missing":
            working_df, msg = fill_missing_text_with_unknown(working_df)
        elif action == "fill_numeric_missing":
            working_df, msg = fill_missing_numeric_with_median(working_df)
        elif action == "drop_rows_many_missing":
            working_df, msg = drop_rows_with_many_missing(working_df)
        else:
            msg = f"Skipped unsupported action: {action}"

        action_messages.append(msg)

    return working_df, action_messages


def make_plan_readable(plan):
    readable_actions = []
    for item in plan.get("actions", []):
        action = item.get("action")
        if action == "remove_duplicates":
            readable_actions.append("Remove exact duplicate rows")
        elif action == "trim_spaces":
            readable_actions.append("Trim extra spaces from text values")
        elif action == "standardize_text_case":
            readable_actions.append("Standardize text case in text columns")
        elif action == "standardize_columns":
            readable_actions.append("Standardize column names")
        elif action == "drop_column":
            readable_actions.append(f"Drop column: {item.get('column_name', 'selected column')}")
        elif action == "fill_text_missing":
            readable_actions.append("Fill missing text values with 'Unknown'")
        elif action == "fill_numeric_missing":
            readable_actions.append("Fill missing numeric values with the median")
        elif action == "drop_rows_many_missing":
            readable_actions.append("Remove rows with many missing values")
    return readable_actions


def suggest_cleaning_actions(df):
    suggestions = []
    duplicate_count = int(df.duplicated().sum())
    total_missing = int(df.isna().sum().sum())
    text_columns = df.select_dtypes(include="object").columns
    numeric_columns = df.select_dtypes(include=["number"]).columns

    has_extra_spaces = False
    has_mixed_case = False

    for column in text_columns:
        series = df[column].dropna().astype(str)
        if len(series) > 0:
            if (series != series.str.strip()).any():
                has_extra_spaces = True
            if (series != series.str.title()).any():
                has_mixed_case = True

    if duplicate_count > 0:
        suggestions.append(f"Remove exact duplicates ({duplicate_count} duplicate rows found).")
    if has_extra_spaces:
        suggestions.append("Trim extra spaces in text columns.")
    if has_mixed_case:
        suggestions.append("Standardize text case in text columns.")
    if any(" " in col or col != col.strip() or col.lower() != col for col in df.columns):
        suggestions.append("Standardize column names.")
    if len(text_columns) > 0 and df[text_columns].isna().sum().sum() > 0:
        suggestions.append("Fill missing text values with 'Unknown'.")
    if len(numeric_columns) > 0 and df[numeric_columns].isna().sum().sum() > 0:
        suggestions.append("Fill missing numeric values with median.")
    if (df.isna().mean(axis=1) * 100 >= 50).any():
        suggestions.append("Review rows with many missing values for possible removal.")
    if total_missing == 0 and duplicate_count == 0 and not has_extra_spaces:
        suggestions.append("No major cleaning issues detected. Review manually for business-specific checks.")
    return suggestions


def calculate_impact(before_df, after_df):
    return {
        "rows_before": len(before_df),
        "rows_after": len(after_df),
        "columns_before": len(before_df.columns),
        "columns_after": len(after_df.columns),
        "missing_before": int(before_df.isna().sum().sum()),
        "missing_after": int(after_df.isna().sum().sum()),
        "duplicates_before": int(before_df.duplicated().sum()),
        "duplicates_after": int(after_df.duplicated().sum()),
        "removed_columns": [col for col in before_df.columns if col not in after_df.columns]
    }


def convert_numeric_like_columns(df):
    converted_df = df.copy()
    converted_cols = []

    for col in converted_df.columns:
        if converted_df[col].dtype == "object":
            series = converted_df[col].astype(str).str.strip()
            converted = pd.to_numeric(series, errors="coerce")
            non_null_original = series.replace({"": np.nan, "nan": np.nan, "None": np.nan}).notna().sum()
            non_null_converted = converted.notna().sum()

            if non_null_original > 0 and (non_null_converted / max(non_null_original, 1)) >= 0.8:
                converted_df[col] = converted
                converted_cols.append(col)

    return converted_df, converted_cols


def plot_bar_counts(series, title, color="#38bdf8"):
    fig, ax = plt.subplots(figsize=(8, 4))
    counts = series.astype(str).value_counts().head(10)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, color=color)
    ax.set_title(title, color="white", fontsize=13, weight="bold")
    ax.set_xlabel("Count", color="white")
    ax.set_ylabel("")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    st.pyplot(fig)


def plot_histogram(series, title, color="#14b8a6"):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(series.dropna(), kde=True, ax=ax, color=color, bins=30)
    ax.set_title(title, color="white", fontsize=13, weight="bold")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    st.pyplot(fig)


def build_model(model_name):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVC": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": MultinomialNB()
    }
    return models[model_name]


def build_preprocessor(X, text_col, model_name):
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    transformers = []

    if text_col != "None" and text_col in X.columns:
        if text_col in categorical_cols:
            categorical_cols.remove(text_col)

        text_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("tfidf", TfidfVectorizer(max_features=2000, stop_words="english"))
        ])
        transformers.append(("text", text_pipeline, text_col))

    if categorical_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

    if numeric_cols:
        if model_name in ["Logistic Regression", "KNN", "Linear SVC"]:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
        else:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])
        transformers.append(("num", num_pipeline, numeric_cols))

    return ColumnTransformer(transformers=transformers)


uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xls", "xlsx", "xlsm"]
)

if uploaded_file is not None:
    try:
        df, file_type = load_uploaded_file(uploaded_file)

        if "file_name" not in st.session_state or st.session_state.file_name != uploaded_file.name:
            st.session_state.file_name = uploaded_file.name
            st.session_state.original_df = df.copy()
            st.session_state.cleaned_df = df.copy()
            st.session_state.actions_applied = []
            st.session_state.last_plan = None
            st.session_state.pending_plan = None
            st.session_state.last_raw_response = ""
            st.session_state.suggested_actions = suggest_cleaning_actions(df)
            st.session_state.preview_impact = None

        current_df = st.session_state.cleaned_df
        st.info(f"Uploaded file: {uploaded_file.name} | Detected type: {file_type.upper()}")

        tab_clean, tab_eda, tab_ml = st.tabs(["Cleaning", "Exploratory Analysis", "Machine Learning"])

        with tab_clean:
            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("Suggested Cleaning Actions")
            for i, suggestion in enumerate(st.session_state.suggested_actions, start=1):
                st.write(f"{i}. {suggestion}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("AI Cleaning Planner")

            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                api_key = ""

            if api_key:
                st.success("Gemini API key found in secrets.toml")
                st.caption("Using model: gemini-3.6-flash")
            else:
                st.warning("Gemini API key not found in .streamlit/secrets.toml")

            user_request = st.text_area(
                "Describe what you want to clean",
                placeholder="Example: drop latitude column, trim spaces, standardize text case"
            )

            c_preview, c_finalize, c_reject = st.columns(3)

            if c_preview.button("Preview AI Plan"):
                if not api_key:
                    st.warning("Please add GEMINI_API_KEY in .streamlit/secrets.toml")
                elif not user_request.strip():
                    st.warning("Please enter a cleaning request.")
                else:
                    try:
                        plan, raw_response = get_api_plan(user_request, list(current_df.columns), api_key)
                        preview_df, _ = run_actions_from_plan(current_df, plan)
                        st.session_state.pending_plan = plan
                        st.session_state.last_raw_response = raw_response
                        st.session_state.preview_impact = calculate_impact(current_df, preview_df)
                        st.success("AI plan created. Review it below before finalizing.")
                    except Exception as e:
                        st.error(f"Could not create AI plan: {e}")

            if st.session_state.pending_plan:
                st.subheader("Proposed Cleaning Plan")
                readable_actions = make_plan_readable(st.session_state.pending_plan)
                for i, action in enumerate(readable_actions, start=1):
                    st.write(f"{i}. {action}")

                if st.session_state.preview_impact:
                    impact = st.session_state.preview_impact
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Rows", impact["rows_after"], impact["rows_after"] - impact["rows_before"])
                    p2.metric("Columns", impact["columns_after"], impact["columns_after"] - impact["columns_before"])
                    p3.metric("Missing Values", impact["missing_after"], impact["missing_after"] - impact["missing_before"])
                    p4.metric("Duplicate Rows", impact["duplicates_after"], impact["duplicates_after"] - impact["duplicates_before"])

                with st.expander("Show technical AI plan"):
                    st.json(st.session_state.pending_plan)

                with st.expander("Show raw Gemini response"):
                    st.code(st.session_state.last_raw_response, language="json")

                if c_finalize.button("Finalize Task"):
                    try:
                        cleaned_df, messages = run_actions_from_plan(current_df, st.session_state.pending_plan)
                        st.session_state.cleaned_df = cleaned_df
                        st.session_state.actions_applied.extend([f"AI plan: {m}" for m in messages])
                        st.session_state.last_plan = st.session_state.pending_plan
                        st.session_state.pending_plan = None
                        st.session_state.last_raw_response = ""
                        st.session_state.suggested_actions = suggest_cleaning_actions(cleaned_df)
                        st.session_state.preview_impact = None
                        st.success("Finalized plan applied successfully.")
                    except Exception as e:
                        st.error(f"Could not finalize the plan: {e}")

                if c_reject.button("Reject Plan"):
                    st.session_state.pending_plan = None
                    st.session_state.last_raw_response = ""
                    st.session_state.preview_impact = None
                    st.info("Plan rejected. No changes were applied.")

            if st.session_state.last_plan:
                st.subheader("Last Finalized AI Plan")
                st.json(st.session_state.last_plan)

            current_df = st.session_state.cleaned_df

            st.subheader("Manual Cleaning Actions")
            selected_actions = st.multiselect(
                "Choose one or more cleaning actions",
                [
                    "Remove Exact Duplicates",
                    "Trim Extra Spaces in Text",
                    "Standardize Text Case",
                    "Standardize Column Names",
                    "Fill Text Missing With Unknown",
                    "Fill Numeric Missing With Median",
                    "Remove Rows With Many Missing Values"
                ]
            )

            if st.button("Apply Selected Cleaning Actions"):
                if not selected_actions:
                    st.warning("Please select at least one cleaning action.")
                else:
                    working_df = current_df.copy()
                    action_messages = []

                    for action in selected_actions:
                        if action == "Remove Exact Duplicates":
                            working_df, msg = remove_exact_duplicates(working_df)
                        elif action == "Trim Extra Spaces in Text":
                            working_df, msg = trim_text_columns(working_df)
                        elif action == "Standardize Text Case":
                            working_df, msg = standardize_text_case(working_df)
                        elif action == "Standardize Column Names":
                            working_df, msg = standardize_column_names(working_df)
                        elif action == "Fill Text Missing With Unknown":
                            working_df, msg = fill_missing_text_with_unknown(working_df)
                        elif action == "Fill Numeric Missing With Median":
                            working_df, msg = fill_missing_numeric_with_median(working_df)
                        elif action == "Remove Rows With Many Missing Values":
                            working_df, msg = drop_rows_with_many_missing(working_df)
                        else:
                            msg = f"Skipped unsupported action: {action}"

                        action_messages.append(msg)

                    st.session_state.cleaned_df = working_df
                    st.session_state.actions_applied.extend(action_messages)
                    st.session_state.suggested_actions = suggest_cleaning_actions(working_df)
                    st.success("Selected cleaning actions applied successfully.")

            current_df = st.session_state.cleaned_df

            st.subheader("Drop Column")
            dc1, dc2 = st.columns([2, 1])

            column_to_drop = dc1.selectbox(
                "Select column to drop",
                options=list(current_df.columns),
                key="drop_column_select"
            )

            if dc2.button("Drop Selected Column", key="drop_column_button"):
                cleaned_df, message = drop_column_by_name(current_df, column_to_drop)
                st.session_state.cleaned_df = cleaned_df
                st.session_state.actions_applied.append(message)
                st.session_state.suggested_actions = suggest_cleaning_actions(cleaned_df)
                st.success(message)

            current_df = st.session_state.cleaned_df
            report_bytes = generate_html_report(
                st.session_state.original_df,
                st.session_state.cleaned_df,
                st.session_state.actions_applied
            )

            st.subheader("Downloads")
            d1, d2, d3 = st.columns(3)
            with d1:
                st.download_button("Download Cleaned CSV", make_downloadable_csv(current_df), "cleaned_data.csv", "text/csv")
            with d2:
                st.download_button(
                    "Download Cleaned Excel",
                    make_downloadable_excel(current_df),
                    "cleaned_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            with d3:
                st.download_button("Download HTML Report", report_bytes, "cleaning_report.html", "text/html")

            st.markdown("</div>", unsafe_allow_html=True)

        with tab_eda:
            analysis_df, converted_cols = convert_numeric_like_columns(st.session_state.cleaned_df)

            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("Exploratory Analysis")

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Rows", analysis_df.shape[0])
            e2.metric("Columns", analysis_df.shape[1])
            e3.metric("Missing Values", int(analysis_df.isna().sum().sum()))
            e4.metric("Duplicate Rows", int(analysis_df.duplicated().sum()))

            if converted_cols:
                st.info(f"Converted numeric-like text columns for analysis: {', '.join(converted_cols)}")

            object_cols = analysis_df.select_dtypes(include=["object"]).columns.tolist()
            numeric_cols = analysis_df.select_dtypes(include=["number"]).columns.tolist()

            c1, c2 = st.columns(2)
            with c1:
                if object_cols:
                    cat_col = st.selectbox("Categorical chart column", object_cols, key="eda_cat")
                    plot_bar_counts(analysis_df[cat_col], f"Top values in {cat_col}")
            with c2:
                if numeric_cols:
                    num_col = st.selectbox("Numeric chart column", numeric_cols, key="eda_num")
                    plot_histogram(analysis_df[num_col], f"Distribution of {num_col}")

            if object_cols:
                target_dist = st.selectbox("Label distribution column", object_cols, key="label_dist")
                plot_bar_counts(analysis_df[target_dist], f"Distribution of {target_dist}", color="#22c55e")

            st.markdown("</div>", unsafe_allow_html=True)

        with tab_ml:
            ml_df_base, converted_cols = convert_numeric_like_columns(st.session_state.cleaned_df)

            st.markdown('<div class="soft-card">', unsafe_allow_html=True)
            st.subheader("Machine Learning")

            all_cols = ml_df_base.columns.tolist()
            object_cols = ml_df_base.select_dtypes(include=["object"]).columns.tolist()

            if len(all_cols) < 2:
                st.warning("Not enough columns available for machine learning.")
            else:
                default_target = all_cols[0]
                churn_cols = [c for c in all_cols if c.lower() == "churn"]
                if churn_cols:
                    default_target = churn_cols[0]

                target_col = st.selectbox(
                    "Select target column",
                    all_cols,
                    index=0 if default_target not in all_cols else all_cols.index(default_target)
                )

                text_options = ["None"] + object_cols
                text_col = st.selectbox("Select text column", text_options)

                ml_mode = st.radio(
                    "Choose ML mode",
                    ["Single Model Analysis", "Compare Multiple Models"],
                    horizontal=True
                )

                sample_size = st.number_input(
                    "Sample rows for training",
                    min_value=100,
                    max_value=max(100, len(ml_df_base)),
                    value=min(3000, len(ml_df_base)),
                    step=100
                )

                available_models = [
                    "Logistic Regression",
                    "Linear SVC",
                    "Random Forest",
                    "Gradient Boosting",
                    "Decision Tree",
                    "KNN",
                    "Naive Bayes"
                ]

                if converted_cols:
                    st.info(f"Converted numeric-like text columns for ML: {', '.join(converted_cols)}")

                if ml_mode == "Single Model Analysis":
                    model_name = st.selectbox("Select one model", available_models)

                    if st.button("Run Single Model Analysis"):
                        try:
                            ml_df = ml_df_base.copy().dropna(subset=[target_col])

                            if len(ml_df) > sample_size:
                                ml_df = ml_df.sample(sample_size, random_state=42)

                            X = ml_df.drop(columns=[target_col])
                            y = ml_df[target_col].astype(str)

                            drop_cols = [c for c in ["order_id", "customer_id", "customerid", "Order ID", "Customer ID"] if c in X.columns]
                            if drop_cols:
                                X = X.drop(columns=drop_cols)

                            label_encoder = LabelEncoder()
                            y_encoded = label_encoder.fit_transform(y)

                            if len(np.unique(y_encoded)) < 2:
                                st.error("Target column must have at least 2 classes.")
                            else:
                                preprocessor = build_preprocessor(X, text_col, model_name)
                                clf = Pipeline([
                                    ("preprocessor", preprocessor),
                                    ("model", build_model(model_name))
                                ])

                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                                )

                                clf.fit(X_train, y_train)
                                y_pred = clf.predict(X_test)

                                acc = accuracy_score(y_test, y_pred)
                                f1 = f1_score(y_test, y_pred, average="weighted")
                                report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
                                cm = confusion_matrix(y_test, y_pred)

                                m1, m2, m3 = st.columns(3)
                                m1.metric("Accuracy", f"{acc:.4f}")
                                m2.metric("F1 Score", f"{f1:.4f}")
                                m3.metric("Test Rows", len(X_test))

                                st.write("#### Classification Report")
                                st.code(report)

                                st.write("#### Confusion Matrix")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.heatmap(
                                    cm,
                                    annot=True,
                                    fmt="d",
                                    cmap="Blues",
                                    xticklabels=label_encoder.classes_,
                                    yticklabels=label_encoder.classes_,
                                    ax=ax
                                )
                                ax.set_xlabel("Predicted")
                                ax.set_ylabel("Actual")
                                ax.set_title(f"Confusion Matrix - {model_name}")
                                fig.patch.set_facecolor("#0f172a")
                                ax.set_facecolor("#0f172a")
                                ax.xaxis.label.set_color("white")
                                ax.yaxis.label.set_color("white")
                                ax.title.set_color("white")
                                ax.tick_params(colors="white")
                                st.pyplot(fig)

                                st.write("#### Model Summary")
                                st.write(f"Model used: {model_name}")
                                st.write(f"Target column: {target_col}")
                                st.write(f"Text column: {text_col}")
                                st.write(f"Training rows: {len(X_train)}")
                                st.write(f"Testing rows: {len(X_test)}")

                        except Exception as e:
                            st.error(f"Single model analysis failed: {e}")

                else:
                    selected_models = st.multiselect(
                        "Select multiple models to compare",
                        available_models,
                        default=["Logistic Regression", "Random Forest", "Linear SVC"]
                    )

                    if st.button("Compare Selected Models"):
                        try:
                            if not selected_models:
                                st.warning("Please select at least one model.")
                            else:
                                ml_df = ml_df_base.copy().dropna(subset=[target_col])

                                if len(ml_df) > sample_size:
                                    ml_df = ml_df.sample(sample_size, random_state=42)

                                X = ml_df.drop(columns=[target_col])
                                y = ml_df[target_col].astype(str)

                                drop_cols = [c for c in ["order_id", "customer_id", "customerid", "Order ID", "Customer ID"] if c in X.columns]
                                if drop_cols:
                                    X = X.drop(columns=drop_cols)

                                label_encoder = LabelEncoder()
                                y_encoded = label_encoder.fit_transform(y)

                                if len(np.unique(y_encoded)) < 2:
                                    st.error("Target column must have at least 2 classes.")
                                else:
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                                    )

                                    results = []

                                    for model_name in selected_models:
                                        preprocessor = build_preprocessor(X, text_col, model_name)
                                        clf = Pipeline([
                                            ("preprocessor", preprocessor),
                                            ("model", build_model(model_name))
                                        ])

                                        clf.fit(X_train, y_train)
                                        y_pred = clf.predict(X_test)

                                        results.append({
                                            "Model": model_name,
                                            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                                            "F1 Score": round(f1_score(y_test, y_pred, average="weighted"), 4)
                                        })

                                    results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
                                    st.dataframe(results_df, use_container_width=True)
                                    st.success(f"Best among selected models: {results_df.iloc['Model']}")

                        except Exception as e:
                            st.error(f"Model comparison failed: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="soft-card">', unsafe_allow_html=True)
        st.subheader("Current Data Preview")
        st.dataframe(st.session_state.cleaned_df.head(), use_container_width=True)

        current_df = st.session_state.cleaned_df
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Rows", len(current_df))
        s2.metric("Columns", len(current_df.columns))
        s3.metric("Missing Values", int(current_df.isna().sum().sum()))
        s4.metric("Exact Duplicate Rows", int(current_df.duplicated().sum()))

        st.subheader("Actions Applied")
        if st.session_state.actions_applied:
            for action in st.session_state.actions_applied:
                st.write(f"- {action}")
        else:
            st.info("No cleaning actions applied yet.")

        if st.button("Reset to Original File"):
            st.session_state.original_df = df.copy()
            st.session_state.cleaned_df = df.copy()
            st.session_state.actions_applied = []
            st.session_state.last_plan = None
            st.session_state.pending_plan = None
            st.session_state.last_raw_response = ""
            st.session_state.suggested_actions = suggest_cleaning_actions(df)
            st.session_state.preview_impact = None
            st.success("Reset complete.")

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Could not read this file: {error}")