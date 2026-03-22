import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

#DATA FRACTION AND SEED

sample_fraction = 1  # Adjust this to test on reduced sample sizes
data = pd.read_csv('Data/Data.csv')
random_state_value=99

input_features = ['Curcumin Solubility', 'Polarity', 'Hildebrand Solubility Parameters',
                  'Dipole Moment', 'Dielectric constants', 'Viscosity', 'delta d', 'delta p', 'delta h']
output_variables = ['EE%', 'DLC%']

X = data[input_features]
y = data[output_variables]

#TRAIN AND TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_value)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Fractional Sampling Logic
if sample_fraction < 1.0:
    X_train_sampled, _, y_train_sampled, _ = train_test_split(
        X_train_scaled, y_train, train_size=sample_fraction, random_state=random_state_value
    )
else:
    X_train_sampled = X_train_scaled
    y_train_sampled = y_train.copy()

#Baseline Defaults & Parameters to Test
# These are the defaults used when a parameter is NOT being actively tested
baseline_params = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'max_iter': 6000,
    'random_state': 42
}

# The specific variations to test one-at-a-time
parameters_to_test = {
    'hidden_layer_sizes': [(50, 25), (100, 50), (150, 75), (100, 100), (200, 100), (300, 150)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd', 'lbfgs'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01],
    'max_iter': [2000, 4000, 6000],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

all_model_results = []

# One-Factor-At-A-Time Optimization 
for target in output_variables:
    print(f"\n{'='*40}\nEvaluating Target: {target}\n{'='*40}")
    
    for param_name, param_values in parameters_to_test.items():
        print(f"\n--- Testing {param_name} ---")
        
        for value in param_values:
            # baseline parameters
            current_params = baseline_params.copy()
            
            # Overwrite the specific parameter we are testing
            current_params[param_name] = value
            
            # testing learning_rate so MUST use SGD solver
            if param_name == 'learning_rate':
                current_params['solver'] = 'sgd'
                
            #Initialize model
            mlp = MLPRegressor(**current_params)
            
            #  5-Fold Cross Validation on Training Data
            cv_scores = cross_val_score(
                mlp, X_train_sampled, y_train_sampled[target], 
                cv=5, scoring='neg_mean_squared_error', n_jobs=-1
            )
            mean_cv_neg_mse = cv_scores.mean()
            
            #  Fit on full training sample and evaluate on Test Data
            mlp.fit(X_train_sampled, y_train_sampled[target])
            y_pred = mlp.predict(X_test_scaled)
            
            test_r2 = r2_score(y_test[target], y_pred)
            test_mse = mean_squared_error(y_test[target], y_pred)
            actual_iterations = mlp.n_iter_
            
            print(f"Value: {value} | CV Neg MSE: {mean_cv_neg_mse:.4f} | Test R²: {test_r2:.4f}")
            
            #  Save results
            all_model_results.append({
                'Target Variable': target,
                'Parameter Tested': param_name,
                'Value': str(value),
                '5-Fold CV Neg MSE': round(mean_cv_neg_mse, 4),
                'Test R2': round(test_r2, 4),
                'Test MSE': round(test_mse, 4),
                'Iterations Needed': actual_iterations
            })

# Save Results
print("\n[Saving Results]")
best_models_df = pd.DataFrame(all_model_results)

os.makedirs("Data", exist_ok=True)
percent_fraction = int(sample_fraction * 100)
file_name = f"{percent_fraction}{random_state_value}_ofat_model_results.csv"
file_path = os.path.join("Data", file_name)

best_models_df.to_csv(file_path, index=False)
print(f"Results successfully saved to {file_path}")
