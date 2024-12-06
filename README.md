# Behavioral-Design-Analysis-Tool
Creating an AI-driven tool for analyzing single-case design data in behavior analytic research. The goal is to develop a system that can help clinicians select and perform the appropriate statistical tests for different design types, leveraging AI to guide this process.

#Features
   Supported Design Types:
   ABA Reversal Design: Analyzes changes in behavior by alternating between baseline and intervention phases.
   Alternating Treatment Design: Compares the effectiveness of two or more interventions.
   Multiple Baseline Design (Pending Integration): Analyzes staggered implementation of interventions across participants, behaviors, or        
   settings.
   
#Planned Design
    Changing Criterion Design: Not yet implemented. This will assess gradual behavior changes based on stepwise modifications to intervention 
    criteria.
#Key Functionalities
    Automated statistical methods: Bayesian analysis, mixed-effects models, randomization tests, logistic regression, and more.
    Data input: Upload preformatted Excel files or manually input data.
    Interactive interface: Built using Streamlit for easy navigation and real-time analysis


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

#Repository Structure
    Behavioral-Design-Analysis-Tool/
    ├── Code/
    │   ├── app.py                   # Main Streamlit application for ABA and Alternating Treatment Designs
    │   ├── Multiple_Baseline.py     # Separate implementation for Multiple Baseline Design
    ├── Dataset/
    │   ├── ABA_reversal.xlsx        # Dataset for ABA Reversal Design
    │   ├── Multiple-baseline.xlsx   # Dataset for Multiple Baseline Design
    │   ├── SCD Hypothetical Datasets.xlsx  # Main dataset for hypothetical SCD scenarios
    ├── Research_Paper/
    │   ├── [Research PDFs]          # Reference papers on statistical methods
    ├── README.md                    # Project documentation (current file)
    ├── requirements.txt             # Required Python libraries
    └── Pilot Implementation.pdf     # Early-stage project documentation


  

#Getting Started


  Prerequisites
    Ensure you have the following installed:
    Python 3.8+  
  Recommended IDE: PyCharm or VSCode
  Installation
    Clone the repository: git clone https://github.com/herin695/Behavioral-Design-Analysis-Tool.git
    cd Behavioral-Design-Analysis-Tool

    
  Install dependencies:
     pip install -r requirements.txt
     
#Running the Application
  For ABA Reversal and Alternating Treatment Designs:
    streamlit run Code/app.py
    
  For Multiple Baseline Design (separate implementation for now):
    streamlit run Code/Multiple_Baseline.py

#Dataset Information
    Main Dataset (SCD Hypothetical Datasets.xlsx):

  Contains generalized examples for SCEDs.
    Used for testing across all supported designs.
    Design-Specific Datasets:
      ABA_reversal.xlsx: Dataset tailored for ABA Reversal analyses.
      Multiple-baseline.xlsx: Dataset designed for Multiple Baseline implementation.
      Ensure the uploaded dataset aligns with the required format. Examples are provided in the Dataset/ folder.

#Future Enhancements
      Integrate the Multiple Baseline design into the main application.
      Implement Changing Criterion Design.
      Improve data visualization and export capabilities.
      Deploy the tool to Streamlit Cloud or AWS for wider accessibility.
      
    
#The app will guide you to:

  Select the design type.
  Upload the dataset.
  Perform the analysis using suggested statistical methods.
  View detailed results and visualizations.








