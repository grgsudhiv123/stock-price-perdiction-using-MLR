import pandas as pd
from model.Regression import MultiLinearRegression, DataScaler


def clean_data(data):
    data['Vol'] = data['Vol'].str.replace(',', '').astype(float)
    X = data[['Open', 'High', 'Low', 'Vol']].values  # Input features as numpy array
    return X


def main():
    # Load and clean training data
    df_train = pd.read_csv("./OHLC.csv")
    X_train = clean_data(df_train)
    Y_train = df_train["Close"].values

    # Initialize scaler and model
    scaler = DataScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = MultiLinearRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(X_scaled, Y_train)

    # Train predictions
    y_train_predicted = model.predict(X_scaled[:5])
    print("Training Predictions:")
    print(y_train_predicted)
    print("Actual Values:")
    print(Y_train[:5])

    # Load and clean test data
    df_test = pd.read_csv("./2020-01-01.csv")
    X_test = clean_data(df_test)
    Y_test = df_test["Close"].values

    # Transform test data using the same scaler
    X_test_scaled = scaler.transform(X_test)

    # Test predictions
    y_predicted = model.predict(X_test_scaled)
    print("Test Predictions:")
    print(y_predicted[:5])
    print("Actual Test Values:")
    print(Y_test[:5])


# main()
#
