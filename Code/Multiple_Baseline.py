import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro, probplot
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load specific sheet based on design type
def load_data(file, sheet_name):
    """Loads data from a specified sheet in an uploaded Excel file."""
    data = pd.read_excel(file, sheet_name=sheet_name)
    return data

# Detect participants and their phases
def detect_participants_and_phases(data):
    """
    Detect participants and their associated phases (e.g., Baseline, Intervention).
    Columns should be named in the format 'P1-Baseline', 'P1-Intervention', etc.
    Returns a dictionary mapping participants to their phase columns.
    """
    participants = {}
    for col in data.columns:
        if "Baseline" in col or "Intervention" in col:
            participant_id = col.split("-")[0]
            if participant_id not in participants:
                participants[participant_id] = {"Baseline": None, "Intervention": None}
            if "Baseline" in col:
                participants[participant_id]["Baseline"] = col
            elif "Intervention" in col:
                participants[participant_id]["Intervention"] = col
    return participants

# Check for normality with visualization
def normality_test_with_visualization(data, phase):
    """Perform Shapiro-Wilk test and generate a histogram and Q-Q plot."""
    stat, p_value = shapiro(data)
    is_normal = p_value > 0.05

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Histogram
    axes[0].hist(data, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title(f"Histogram of {phase}")
    axes[0].set_xlabel("Values")
    axes[0].set_ylabel("Frequency")

    # Q-Q Plot
    probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot of {phase}")
    plt.tight_layout()

    return is_normal, p_value, fig

# Logistic Regression Analysis
def logistic_regression_analysis(baseline, intervention):
    """Perform logistic regression for the given baseline and intervention data."""
    baseline_labels = [0] * len(baseline)
    intervention_labels = [1] * len(intervention)

    # Combine data and labels
    X = np.concatenate([baseline, intervention]).reshape(-1, 1)
    y = np.array(baseline_labels + intervention_labels)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    return model, report, accuracy, predictions

# Mixed-Effects Model Analysis
def mixed_effects_model_analysis(baseline, intervention):
    """Perform mixed-effects model analysis for the given baseline and intervention data."""
    combined_data = pd.DataFrame({
        "value": pd.concat([baseline, intervention]),
        "phase": ["Baseline"] * len(baseline) + ["Intervention"] * len(intervention),
        "subject": list(range(len(baseline))) + list(range(len(intervention)))
    })
    model = MixedLM.from_formula("value ~ phase", groups="subject", data=combined_data)
    result = model.fit()
    params = result.params.to_dict()
    pvalues = result.pvalues.to_dict()
    return params, pvalues

# Display Mixed-Effects Results
def display_mixed_effects_results(params, pvalues):
    """Display mixed-effects model results in a structured format."""
    st.write("### Mixed-Effects Model Results")

    # Create a DataFrame for better visualization
    results_df = pd.DataFrame({
        "Parameter": list(params.keys()),
        "Estimate": list(params.values()),
        "P-value": [f"{p:.4e}" if not pd.isna(p) else "N/A" for p in pvalues.values()]
    })

    # Highlight significance
    results_df["Significant"] = results_df["P-value"].apply(
        lambda x: "Yes" if x != "N/A" and float(x) < 0.05 else "No"
    )

    st.dataframe(results_df)

    # Provide interpretation
    st.write("### Interpretation")
    if any(results_df["Significant"] == "Yes"):
        st.write("- **Significant Results**: Parameters with significant p-values indicate meaningful effects in the data.")
    else:
        st.write("- **No Significant Results**: No parameters showed significant effects, suggesting limited variability between phases.")

# Randomization Test
def randomization_test(baseline, intervention, n_iterations=1000):
    """Perform randomization test for baseline and intervention data."""
    combined_data = np.concatenate((baseline, intervention))
    observed_diff = np.mean(intervention) - np.mean(baseline)
    randomized_diffs = []

    for _ in range(n_iterations):
        np.random.shuffle(combined_data)
        shuffled_baseline = combined_data[:len(baseline)]
        shuffled_intervention = combined_data[len(baseline):]
        randomized_diff = np.mean(shuffled_intervention) - np.mean(shuffled_baseline)
        randomized_diffs.append(randomized_diff)

    p_value = np.mean(np.abs(randomized_diffs) >= np.abs(observed_diff))
    return observed_diff, p_value

# Streamlit UI for File Upload and Analysis
st.title("AI-Powered Statistical Analysis Tool")
st.write("Upload your dataset and perform statistical analysis for participants or phases.")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")
if uploaded_file:
    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    selected_sheet = st.selectbox("Select the sheet", sheet_names)
    data = load_data(uploaded_file, selected_sheet)

    st.write("### Data Preview")
    st.dataframe(data.head())

    # Detect participants and phases
    participants = detect_participants_and_phases(data)
    if not participants:
        st.error("No participants or phases detected. Ensure your columns follow the format 'P1-Baseline', 'P1-Intervention', etc.")
    else:
        st.write("### Detected Participants and Phases")
        st.json(participants)

        selected_participants = st.multiselect(
            "Select participants to analyze",
            list(participants.keys()),
            default=list(participants.keys())
        )

        for participant in selected_participants:
            phases = participants[participant]
            baseline_col = phases["Baseline"]
            intervention_col = phases["Intervention"]

            if baseline_col is None or intervention_col is None:
                st.warning(f"Skipping {participant} due to missing data.")
                continue

            baseline_data = data[baseline_col].dropna()
            intervention_data = data[intervention_col].dropna()

            st.write(f"### Analysis for Participant {participant}")

            # Normality Test with Visualization
            is_normal_baseline, p_baseline, fig_baseline = normality_test_with_visualization(baseline_data, "Baseline")
            st.pyplot(fig_baseline)

            is_normal_intervention, p_intervention, fig_intervention = normality_test_with_visualization(intervention_data, "Intervention")
            st.pyplot(fig_intervention)

            # Logistic Regression Analysis
            model, report, accuracy, predictions = logistic_regression_analysis(baseline_data, intervention_data)
            st.write("### Logistic Regression Results")
            st.write(f"- **Accuracy**: {accuracy * 100:.2f}%")
            st.dataframe(pd.DataFrame(report).transpose())

            # Mixed-Effects Model Analysis
            params, pvalues = mixed_effects_model_analysis(baseline_data, intervention_data)
            display_mixed_effects_results(params, pvalues)

            # Randomization Test
            observed_diff, p_value = randomization_test(baseline_data, intervention_data)
            st.write("### Randomization Test Results")
            st.write(f"- **Observed Difference**: {observed_diff:.4f}")
            st.write(f"- **P-value**: {p_value:.4f}")
