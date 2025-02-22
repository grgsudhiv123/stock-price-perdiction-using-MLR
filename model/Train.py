import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from model.Regression import MultiLinearRegression, DataScaler

def clean_data(data):
    # Clean the 'Vol' column and extract features
    # data['Vol'] = data['Vol'].str.replace(',', '').astype(float)
    # data['Vol'] = data['Vol'].str.replace(',', '').astype(float)
    X = data[['Open', 'High', 'Low', 'Vol']].values  # Return as NumPy array
    return X

def train_test_evaluation(X, Y):
    # Split data into training and testing sets (80/20 split)
    split_index = int(0.8 * len(X))
    X_train_part = X[:split_index]
    Y_train_part = Y[:split_index]
    X_test_part = X[split_index:]
    Y_test_part = Y[split_index:]
    
    # Scale the data
    scaler = DataScaler()
    X_train_scaled = scaler.fit_transform(X_train_part)
    X_test_scaled = scaler.transform(X_test_part)
    
    # Train the model
    model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_train_scaled, Y_train_part)
    
    # Predict on test set and calculate RMSE
    Y_test_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(Y_test_part, Y_test_pred)
    rmse = np.sqrt(mse)  # Compute RMSE from MSE
    print("Train-Test Split Evaluation:")
    print("Test RMSE:", rmse)
    return scaler, model

def kfold_evaluation(X, Y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        Y_train_fold, Y_val_fold = Y[train_idx], Y[val_idx]
        
        # Scale the data for this fold
        scaler_fold = DataScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler_fold.transform(X_val_fold)
        
        # Train the model on the fold
        model_fold = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
        model_fold.fit(X_train_fold_scaled, Y_train_fold)
        
        # Predict on the validation fold and compute RMSE
        Y_val_pred = model_fold.predict(X_val_fold_scaled)
        mse_fold = mean_squared_error(Y_val_fold, Y_val_pred)
        rmse_fold = np.sqrt(mse_fold)
        rmse_scores.append(rmse_fold)
        print("Fold RMSE:", rmse_fold)
    
    avg_rmse = np.mean(rmse_scores)
    print("Average RMSE across", k, "folds:", avg_rmse)
    return avg_rmse

def final_model_training(X, Y):
    # Train on the full dataset
    scaler = DataScaler()
    X_scaled = scaler.fit_transform(X)
    final_model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
    final_model.fit(X_scaled, Y)
    
    # Optionally, save the scaler and model using pickle
    # import pickle
    # with open('scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(final_model, f)
    
    print("Final model trained on full dataset.")
    return scaler, final_model

def main():
    # Load and clean full dataset
    df_train = pd.read_csv("./NABIL.csv")
    X = clean_data(df_train)
    Y = df_train["Close"].values
    
    # Perform evaluation using train-test split
    print("\n=== Train-Test Evaluation ===")
    scaler_tt, model_tt = train_test_evaluation(X, Y)
    
    # Perform K-Fold Cross-Validation evaluation
    print("\n=== K-Fold Cross-Validation Evaluation ===")
    avg_rmse = kfold_evaluation(X, Y, k=5)
    
    # Train the final model on the complete dataset
    print("\n=== Final Model Training ===")
    final_scaler, final_model = final_model_training(X, Y)
    
if __name__ == "__main__":
    main()