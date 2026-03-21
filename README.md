# ML-Based Prediction of Drug Loading in Polymer Nanoparticles: Reduced Dataset Optimization

## Overview
This repository contains a streamlined machine learning workflow based on the study: *“Predictive Modelling of Solvent Effects on Drug Incorporation into Polymeric Nanocarriers.”* While the original study utilized a full dataset to predict Encapsulation Efficiency (EE%) and Drug Loading Capacity (DLC%), **this project specifically investigates how model performance scales when training data is restricted.** By reducing the training dataset to specific fractions, we perform hyperparameter optimization to isolate the best Multilayer Perceptron (MLP) configurations for predicting polymer nanoparticle performance under data-constrained conditions.

## Dataset Description
The dataset represents specific solvent conditions used for nanoparticle formulation.
* **Input Features (Solvent Properties):** Curcumin solubility, Hansen solubility parameters (dispersion, polar, hydrogen bonding), Hildebrand solubility parameter, Dielectric constant, Viscosity, Dipole moment, and Polarity.
* **Target Outputs:** **EE%** (Encapsulation Efficiency) and **DLC%** (Drug Loading Capacity).

## Methodology: Fractional Sampling & Optimization
To understand how the MLP model handles limited data, this script systematically trains the model on restricted subsets using a One-Factor-At-A-Time (OFAT) approach:

1. **Fractional Data Sampling:** The training dataset is reduced to a user-defined percentage (e.g., 100%, 80%, 50%), while the test dataset is kept strictly constant to ensure fair, apples-to-apples evaluation across all models.
2. **Standardization:** Features are scaled using `StandardScaler` fitted only on the training subset to prevent data leakage.
3. **One-Factor-At-A-Time (OFAT) Tuning:** Against a default baseline model, we isolate and independently modify hyperparameters including:
   * Hidden layer sizes
   * Activation functions (`relu`, `tanh`, `logistic`)
   * Solvers (`adam`, `sgd`, `lbfgs`)
   * L2 penalty (`alpha`)
   * Maximum iterations and learning rates
4. **Cross-Validation & Evaluation:** Each parameter change is evaluated using 5-Fold Cross-Validation on the restricted training set (scored via Negative Mean Squared Error) and finally tested against the unseen test set to generate R² and MSE metrics.

## Requirements
This project supports Python 3.7 through 3.10. Install the required dependencies using:
`pip install numpy pandas scikit-learn`

*(Note: If you plan to expand this to include the data visualization and SHAP analysis from the original paper, you will also need `matplotlib`, `seaborn`, and `shap`).*

## Output and Interpretation 
When the script finishes running, it automatically generates a detailed CSV report of the optimization process saved in the `Data/` directory. 

The output file is dynamically named based on the data fraction and random state (e.g., `10099_ofat_model_results.csv` for a 100% sample with random state 99). 

**The output logs:**
* The specific target being predicted (EE% or DLC%).
* The hyperparameter being tested and its value.
* The 5-Fold CV Negative MSE (to show how well it trained).
* The final Test R² and Test MSE (to show how well it generalizes).
* The number of iterations the model needed to converge.

By comparing these CSV outputs across different sample fractions, you can pinpoint exactly which hyperparameters are hyper-optimized for smaller datasets versus larger ones.
