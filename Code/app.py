import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal, ttest_rel, wilcoxon, shapiro, beta as beta_dist
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
import scipy.stats as stats


# Load specific sheet based on design type
def load_data(file, sheet_name):
    """Loads data from a specified sheet in an uploaded Excel file."""
    data = pd.read_excel(file, sheet_name=sheet_name)
    return data


# Check for normal distribution
def check_normality(data):
    """Check if the data is normally distributed using the Shapiro-Wilk test."""
    stat, p_value = shapiro(data)
    return {
        "is_normal": p_value > 0.05,  # If p-value > 0.05, data is normal
        "p_value": p_value
    }


# Handle unequal length arrays
def handle_unequal_lengths(data_conditions):
    """Ensures all arrays are of equal length by trimming or filling missing values."""
    min_length = min(len(cond) for cond in data_conditions)
    trimmed_conditions = [cond[:min_length] for cond in data_conditions]
    return trimmed_conditions


# Improved detection of design type
def detect_design_type(sheet_name, data):
    """Improved detection logic for design type."""
    # Check for keywords in the sheet name
    if "ABA" in sheet_name or "Reversal" in sheet_name:
        return "ABA Reversal"
    elif "Alternating" in sheet_name or "Treatment" in sheet_name:
        return "Alternating Treatment"

    # Check for specific column names
    columns_lower = [col.lower() for col in data.columns]
    if any(col in columns_lower for col in ["baseline", "intervention", "return to baseline"]):
        return "ABA Reversal"
    elif len(data.columns) > 3:
        return "Alternating Treatment"

    # If detection is ambiguous, ask the user
    return st.selectbox("Could not auto-detect. Please select the design type:",
                        ["ABA Reversal", "Alternating Treatment"])


# Suggest methods based on the design type and normality
def suggest_methods(design_type, is_normal):
    """Suggest methods based on design type and data distribution (normal/non-normal)."""
    if design_type == "ABA Reversal":
        if is_normal:
            return ["Paired t-test", "Mixed-Effects Models", "Bayesian Analysis"]
        else:
            return ["Wilcoxon Signed-Rank Test", "Mixed-Effects Models", "Bayesian Analysis"]
    elif design_type == "Alternating Treatment":
        if is_normal:
            return ["ANOVA", "Randomization Test"]
        else:
            return ["Kruskal-Wallis Test", "Randomization Test", "Monte Carlo Simulation"]
    return []


# Visualize normality
def visualize_normality(data, title):
    """Generate histogram and Q-Q plot for normality visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(data, bins=20, color='skyblue', edgecolor='black')
    axes[0].set_title(f"Histogram of {title}")
    axes[0].set_xlabel("Values")
    axes[0].set_ylabel("Frequency")

    # Q-Q Plot
    stats.probplot(data, dist="norm", plot=axes[1])
    axes[1].set_title(f"Q-Q Plot of {title}")

    # Adjust layout
    plt.tight_layout()

    return fig


# ABA Reversal Analysis
def aba_reversal_analysis(data_baseline, data_intervention, method):
    """Perform the selected analysis for ABA Reversal, ensuring equal lengths."""
    min_length = min(len(data_baseline), len(data_intervention))
    data_baseline = data_baseline[:min_length]
    data_intervention = data_intervention[:min_length]
    result = {}

    if method == "Paired t-test":
        stat, p_value = ttest_rel(data_baseline, data_intervention, nan_policy='omit')
        result = {"method": method, "statistic": stat, "p_value": p_value}
    elif method == "Wilcoxon Signed-Rank Test":
        stat, p_value = wilcoxon(data_baseline, data_intervention)
        result = {"method": method, "statistic": stat, "p_value": p_value}
    elif method == "Bayesian Analysis":
        prior_alpha, prior_beta = 1, 1
        success = (data_intervention > data_baseline).sum()
        trials = len(data_intervention)
        posterior_alpha = prior_alpha + success
        posterior_beta = prior_beta + trials - success
        posterior_distribution = beta_dist(posterior_alpha, posterior_beta)
        posterior_mean = posterior_distribution.mean()
        result = {"method": method, "posterior_mean": posterior_mean}
    elif method == "Mixed-Effects Models":
        combined_data = pd.DataFrame({
            "value": pd.concat([data_baseline, data_intervention]),
            "phase": ["Baseline"] * len(data_baseline) + ["Intervention"] * len(data_intervention),
            "subject": list(range(len(data_baseline))) + list(range(len(data_intervention)))})
        model = MixedLM.from_formula("value ~ phase", groups="subject", data=combined_data)
        mixed_model_result = model.fit()
        result = {
            "method": method,
            "coef": mixed_model_result.params.to_dict(),
            "pvalues": mixed_model_result.pvalues.to_dict()
        }
    return result


# Alternating Treatment Analysis
def alternating_treatment_analysis(data_conditions, method):
    """Perform the selected analysis for Alternating Treatment, ensuring equal lengths."""
    data_conditions = handle_unequal_lengths(data_conditions)
    result = {}

    if method == "ANOVA":
        stat, p_value = f_oneway(*data_conditions)
        result = {"method": method, "statistic": stat, "p_value": p_value}
    elif method == "Kruskal-Wallis Test":
        stat, p_value = kruskal(*data_conditions)
        result = {"method": method, "statistic": stat, "p_value": p_value}
    elif method == "Randomization Test":
        observed_diff = np.mean(data_conditions[0]) - np.mean(data_conditions[1])
        combined_data = np.concatenate(data_conditions)
        n_iterations = 1000
        sim_diffs = []
        for _ in range(n_iterations):
            np.random.shuffle(combined_data)
            sim_diff = np.mean(combined_data[:len(data_conditions[0])]) - np.mean(
                combined_data[len(data_conditions[0]):])
            sim_diffs.append(sim_diff)
        p_value = np.mean(np.abs(sim_diffs) >= np.abs(observed_diff))
        result = {"method": method, "observed_diff": observed_diff, "p_value": p_value}
    elif method == "Monte Carlo Simulation":
        observed_diff = np.mean(data_conditions[0]) - np.mean(data_conditions[1])
        n_simulations = 1000
        sim_diffs = []
        for _ in range(n_simulations):
            sim_data1 = np.random.choice(data_conditions[0], len(data_conditions[0]), replace=True)
            sim_data2 = np.random.choice(data_conditions[1], len(data_conditions[1]), replace=True)
            sim_diff = np.mean(sim_data1) - np.mean(sim_data2)
            sim_diffs.append(sim_diff)
        p_value = np.mean(np.abs(sim_diffs) >= np.abs(observed_diff))
        result = {"method": method, "observed_diff": observed_diff, "p_value": p_value}
    return result


# Streamlit UI
st.title("AI-Powered Statistical Analysis Tool")
st.subheader("Instructions for File Upload")
st.write("1. Ensure your dataset is in **Excel format (.xlsx)**.")
st.write("2. Include appropriate column names (e.g., 'Baseline', 'Intervention', etc.).")
st.write("3. If unsure about naming conventions, consult the data preview after uploading.")

uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file:
    # Load and preview the data
    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    selected_sheet = st.selectbox("Select the sheet", sheet_names)
    data = load_data(uploaded_file, selected_sheet)
    st.write("Data Preview:", data.head())

    # Detect the design type automatically
    design_type = detect_design_type(selected_sheet, data)
    st.write(f"Detected Design Type: **{design_type}**")

    if design_type == "ABA Reversal":
        st.write("### ABA Reversal Analysis")
        baseline_cols = st.multiselect("Select Baseline Columns", data.columns, default=data.columns[:1])
        intervention_cols = st.multiselect("Select Intervention Columns", data.columns, default=data.columns[1:2])

        # Combine selected columns
        baseline_data = data[baseline_cols].mean(axis=1).dropna()
        intervention_data = data[intervention_cols].mean(axis=1).dropna()

        # Automatically check normality
        normality_baseline = check_normality(baseline_data)
        normality_intervention = check_normality(intervention_data)
        is_normal = normality_baseline["is_normal"] and normality_intervention["is_normal"]

        st.write("### Normality Test Results")
        st.write(f"- **Baseline Normality**: {'Normal' if normality_baseline['is_normal'] else 'Not Normal'} (p-value: {normality_baseline['p_value']:.4e})")
        st.write(f"- **Intervention Normality**: {'Normal' if normality_intervention['is_normal'] else 'Not Normal'} (p-value: {normality_intervention['p_value']:.4e})")

        # Visualization
        st.write("### Visualize Normality")
        st.pyplot(visualize_normality(baseline_data, "Baseline"))
        st.pyplot(visualize_normality(intervention_data, "Intervention"))

        # Suggest methods based on normality
        suggested_methods = suggest_methods("ABA Reversal", is_normal)
        method = st.selectbox("Select Statistical Method", suggested_methods)

        # Perform the analysis
        result = aba_reversal_analysis(baseline_data, intervention_data, method)

        # Display results
        st.write(f"### {result.get('method', 'Analysis')} Results:")
        if method == "Mixed-Effects Models":
            # Improved formatting for Mixed-Effects Models
            st.write("**Coefficients:**")
            for coef, val in result["coef"].items():
                st.write(f"- {coef}: {val:.4f}")

            st.write("**P-values:**")
            for pval, val in result["pvalues"].items():
                if not pd.isna(val):
                    st.write(f"- {pval}: {val:.4e}")
                else:
                    st.write(f"- {pval}: Not Applicable (N/A)")
        else:
            for key, value in result.items():
                if key not in ["method"]:
                    st.write(f"- {key.capitalize()}: {value}")

        # Feedback Section
        st.write("### Interpretation and Feedback")
        if method in ["Paired t-test", "Wilcoxon Signed-Rank Test"]:
            if result["p_value"] < 0.05:
                st.write("  - Interpretation: Statistically significant, suggesting the intervention had an effect.")
            else:
                st.write("  - Interpretation: Not statistically significant, indicating no clear effect.")
        elif method == "Bayesian Analysis":
            st.write(f"- **Posterior Mean**: {result['posterior_mean']:.4f}")
            if result["posterior_mean"] > 0.5:
                st.write("  - Interpretation: The intervention likely had a positive effect.")
            else:
                st.write("  - Interpretation: The intervention likely had little or no effect.")
        elif method == "Mixed-Effects Models":
            intercept_pval = result["pvalues"].get("Intercept", None)
            intervention_pval = result["pvalues"].get("phase[T.Intervention]", None)

            if intercept_pval is not None and intercept_pval < 0.05:
                st.write("- **Baseline (Intercept):** Statistically significant, suggesting a meaningful baseline effect.")
            else:
                st.write("- **Baseline (Intercept):** Not statistically significant, suggesting no strong baseline effect.")

            if intervention_pval is not None and intervention_pval < 0.05:
                st.write("- **Intervention Effect:** Statistically significant, indicating a strong effect of the intervention.")
            else:
                st.write("- **Intervention Effect:** Not statistically significant, indicating no strong evidence of an intervention effect.")

    elif design_type == "Alternating Treatment":
        st.write("### Alternating Treatment Analysis")
        condition_cols = st.multiselect("Select Columns for Conditions", data.columns)

        if len(condition_cols) < 2:
            st.write("Please select at least two columns for the analysis.")
        else:
            data_conditions = [data[col].dropna().values for col in condition_cols]

            # Check normality for all conditions
            normality_results = [check_normality(cond) for cond in data_conditions]
            all_normal = all(res["is_normal"] for res in normality_results)

            st.write("### Normality Test Results for Each Condition")
            for i, res in enumerate(normality_results):
                st.write(f"- **Condition {i + 1} Normality**: {'Normal' if res['is_normal'] else 'Not Normal'} (p-value: {res['p_value']:.4e})")

            # Visualization
            st.write("### Visualize Normality for Each Condition")
            for i, cond in enumerate(data_conditions):
                st.pyplot(visualize_normality(cond, f"Condition {i + 1}"))

            # Suggest methods based on normality
            suggested_methods = suggest_methods("Alternating Treatment", all_normal)
            method = st.selectbox("Select Statistical Method", suggested_methods)

            # Perform the analysis
            result = alternating_treatment_analysis(data_conditions, method)

            # Display results
            st.write(f"### {result.get('method', 'Analysis')} Results:")
            for key, value in result.items():
                if key not in ["method"]:
                    st.write(f"- {key.capitalize()}: {value}")

            # Feedback Section
            st.write("### Interpretation and Feedback")
            if method == "ANOVA":
                if result["p_value"] < 0.05:
                    st.write("  - Interpretation: Statistically significant differences exist between conditions.")
                    st.write("  - Suggestion: Perform post-hoc analysis to identify specific differences between groups.")
                else:
                    st.write("  - Interpretation: No statistically significant differences found between conditions.")
                    st.write("  - Suggestion: Consider increasing the sample size or reviewing the experimental design.")
            elif method == "Kruskal-Wallis Test":
                if result["p_value"] < 0.05:
                    st.write("  - Interpretation: Statistically significant differences exist between conditions.")
                    st.write("  - Suggestion: Explore the specific factors driving these differences.")
                else:
                    st.write("  - Interpretation: No statistically significant differences found.")
                    st.write("  - Suggestion: Increase sample size or check data variability.")
            elif method in ["Randomization Test", "Monte Carlo Simulation"]:
                if result["p_value"] < 0.05:
                    st.write("  - Interpretation: Statistically significant differences exist between conditions.")
                    st.write("  - Suggestion: Randomization or simulation results suggest a robust effect; consider replicating to confirm findings.")
                else:
                    st.write("  - Interpretation: No statistically significant differences found.")
                    st.write("  - Suggestion: Increase the number of simulations or evaluate data consistency.")

