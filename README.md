# Behavioral-Design-Analysis-Tool
Creating an AI-driven tool for analyzing single-case design data in behavior analytic research. The goal is to develop a system that can help clinicians select and perform the appropriate statistical tests for different design types, leveraging AI to guide this process.

#Features
The tool currently supports the following designs:

#ABA Reversal Design
  Analyze baseline, intervention, and return-to-baseline phases.

  Statistical methods:
    Paired t-test
    Wilcoxon Signed-Rank Test
    Mixed-Effects Models
    Bayesian Analysis
    
#Alternating Treatment Design

Compare performance across multiple treatment conditions.

  Statistical methods:
    ANOVA
    Kruskal-Wallis Test
    Randomization Test
    Monte Carlo Simulation
    
#Multiple Baseline Design
  Analyze treatment effects across different subjects or settings over time.

  Statistical methods:
    Logistic Regression
    Wilcoxon Signed-Rank Test
    Randomization Test
    
#Future Work
  The project is designed to allow future contributors to implement the following designs: 4. Changing Criterion.


#Requirements
  Python Dependencies
  Python >= 3.7
  Libraries:
  numpy
  pandas
  matplotlib
  scipy
  statsmodels
  streamlit
  scikit-learn
  To install all dependencies, run:

#pip install -r requirements.txt


#Setup and Usage
  Clone the Repository
  git clone https://github.com/username/Behavioral-Design-Analysis-Tool


#Install Dependencies
  Ensure Python is installed on your system. Then, run:


  pip install -r requirements.txt
  Run the Application
#Launch the Streamlit app by executing:

  streamlit run filename.py
  
#Upload a Dataset
  Prepare an Excel file (.xlsx) with appropriate column names for your experimental design:

ABA Reversal Design: Columns such as Baseline and Intervention.
Alternating Treatment Design: Columns for each treatment condition.
Multiple Baseline Design: Columns for multiple participants or settings, with phases clearly labeled (e.g., P1-Baseline, P1-Intervention).
The app will guide you to:

Select the design type.
Upload the dataset.
Perform the analysis using suggested statistical methods.
View detailed results and visualizations.








