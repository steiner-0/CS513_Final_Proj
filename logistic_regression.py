import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath='output/flight_weather_merged.csv'):
    """Load the flight-weather merged data"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

def prepare_data(df):
    """Prepare data for logistic regression"""
    print("Preparing data for modeling...")
    
    # Create binary target variable - flight has weather delay or not
    df['has_delay'] = (df['WEATHER_DELAY'] > 0).astype(int)
    
    # Select only numeric weather-related features at origin airport
    # We're focusing on origin airport as per the analysis it has more impact
    weather_features = [
        'origin_temp', 'origin_dwpt', 'origin_rhum', 'origin_prcp', 
        'origin_snow', 'origin_wspd', 'origin_wpgt', 'origin_pres', 'origin_coco'
    ]
    
    # Keep only rows with no missing values in these features
    model_df = df.dropna(subset=weather_features + ['has_delay'])
    
    print(f"After filtering, shape: {model_df.shape}")
    print(f"Positive class (delayed flights) percentage: {model_df['has_delay'].mean() * 100:.2f}%")
    
    # Prepare features and target
    X = model_df[weather_features]
    y = model_df['has_delay']
    
    # Standardize numeric features (except coco which is categorical)
    scaler = StandardScaler()
    numeric_features = [f for f in weather_features if f != 'origin_coco']
    X[numeric_features] = scaler.fit_transform(X[numeric_features])
    
    # Add coco as one-hot encoded features
    # We do this to capture the categorical nature of weather conditions
    X_with_coco = pd.get_dummies(X, columns=['origin_coco'], prefix='coco')
    
    return X_with_coco, y, model_df

def build_and_evaluate_model(X, y):
    """Build and evaluate a logistic regression model"""
    print("Building and evaluating the model...")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Create and train the model
    # Using a higher max_iter as the default might not be enough for convergence
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.coef_[0])
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    return model, feature_importance

def plot_feature_importance(feature_importance):
    """Plot feature importance"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Features for Predicting Weather Delays')
    plt.tight_layout()
    plt.savefig('logistic_regression_feature_importance.png')
    print("Feature importance plot saved as 'logistic_regression_feature_importance.png'")

def analyze_coco_impact(model, X):
    """Analyze the impact of different coco values on delay prediction"""
    # Get coco-related coefficients
    coco_features = [col for col in X.columns if col.startswith('coco_')]
    coco_coeffs = {f.replace('coco_', ''): coef for f, coef in zip(coco_features, model.coef_[0][-(len(coco_features)):]) }
    
    # Sort by coefficient value (positive = more likely to cause delay)
    sorted_cocos = sorted(coco_coeffs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nWeather Condition Impact on Delay Probability (from most to least):")
    for coco, coef in sorted_cocos:
        print(f"COCO {coco}: {coef:.4f}")

def main():
    """Main function to run the analysis"""
    # Load the data
    df = load_data()
    
    # Prepare the data
    X, y, model_df = prepare_data(df)
    
    # Build and evaluate the model
    model, feature_importance = build_and_evaluate_model(X, y)
    
    # Plot feature importance
    plot_feature_importance(feature_importance)
    
    # Analyze coco impact
    analyze_coco_impact(model, X)
    
    print("\nLogistic regression analysis complete.")

if __name__ == "__main__":
    main()