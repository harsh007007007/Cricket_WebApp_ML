import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

class CricketPerformanceModel:
    def __init__(self):
        # Preprocessing
        self.scaler = StandardScaler()
        # Clustering model
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        # Regression models
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)
        self.gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        self.lr = LinearRegression()
        # Keep models in a dict to iterate later
        self.models = {
            "Random Forest": self.rf,
            "Gradient Boosting": self.gb,
            "Linear Regression": self.lr
        }
        # Similarity lookup (for similar players)
        self.similarity = NearestNeighbors(n_neighbors=6)

    def clean_data(self, df):
        """Convert columns to numeric, handling dashes and non-numeric values."""
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce')
        return df.fillna(0)
    
    def prepare_features(self, df):
        """Feature engineering from raw data."""
        df = self.clean_data(df.copy())
        features = pd.DataFrame()
        # Create batting features
        for fmt in ['Tests', 'ODIs', 'T20s']:
            features[f'bat_{fmt}_avg'] = df[f'BATTING_{fmt}_Ave']
            features[f'bat_{fmt}_sr'] = df[f'BATTING_{fmt}_SR']
        # Create bowling features
        for fmt in ['Tests', 'ODIs', 'T20s']:
            features[f'bowl_{fmt}_avg'] = df[f'BOWLING_{fmt}_Ave']
            features[f'bowl_{fmt}_econ'] = df[f'BOWLING_{fmt}_Econ']
        return features

    def train_models(self, df):
        """Train all models using cleaned data."""
        features = self.prepare_features(df)
        X = self.scaler.fit_transform(features)
        # Train clustering model
        self.kmeans.fit(X)
        # Use Test Runs as target for regression
        y = df['BATTING_Tests_Runs'].values
        for name, model in self.models.items():
            model.fit(X, y)
        # Train similarity model
        self.similarity.fit(X)
        return self

    def predict_cluster(self, features):
        """Predict which cluster a player belongs to."""
        X = self.scaler.transform(features)
        return self.kmeans.predict(X)[0]
    
    def predict_performance(self, features):
        """Ensemble prediction: returns predictions from each model,
        the ensemble (average) prediction, and a confidence score."""
        X = self.scaler.transform(features)
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)[0]
        # Ensemble prediction (average)
        pred_values = np.array(list(predictions.values()))
        ensemble_pred = float(np.mean(pred_values))
        ensemble_std = float(np.std(pred_values))
        # A simple confidence metric: lower variance means higher confidence.
        confidence = 1 - (ensemble_std / (ensemble_pred + 1e-6))
        return {"ensemble": ensemble_pred, "confidence": confidence, "individual": predictions}

    def get_top_performers(self, format_type, df):
        """Return top performers (batsmen and bowlers) by sorting the DataFrame."""
        if format_type == 'test':
            df_sorted_batsmen = df.dropna(subset=['BATTING_Tests_Runs']).sort_values('BATTING_Tests_Runs', ascending=False)
            top_batsmen_df = df_sorted_batsmen.head(10)
            df_sorted_bowlers = df.dropna(subset=['BOWLING_Tests_Wkts']).sort_values('BOWLING_Tests_Wkts', ascending=False)
            top_bowlers_df = df_sorted_bowlers.head(10)
        elif format_type == 'odi':
            df_sorted_batsmen = df.dropna(subset=['BATTING_ODIs_Runs']).sort_values('BATTING_ODIs_Runs', ascending=False)
            top_batsmen_df = df_sorted_batsmen.head(10)
            df_sorted_bowlers = df.dropna(subset=['BOWLING_ODIs_Wkts']).sort_values('BOWLING_ODIs_Wkts', ascending=False)
            top_bowlers_df = df_sorted_bowlers.head(10)
        elif format_type == 't20':
            df_sorted_batsmen = df.dropna(subset=['BATTING_T20Is_Runs']).sort_values('BATTING_T20Is_Runs', ascending=False)
            top_batsmen_df = df_sorted_batsmen.head(10)
            df_sorted_bowlers = df.dropna(subset=['BOWLING_T20Is_Wkts']).sort_values('BOWLING_T20Is_Wkts', ascending=False)
            top_bowlers_df = df_sorted_bowlers.head(10)
        top_batsmen = top_batsmen_df.to_dict('records')
        top_bowlers = top_bowlers_df.to_dict('records')
        return {"batsmen": top_batsmen, "bowlers": top_bowlers}
    
    def get_model_descriptions(self):
        """Return descriptions for each model."""
        return {
            "Random Forest": "An ensemble of decision trees that models non-linear interactions, provides robust predictions and computes feature importance.",
            "Gradient Boosting": "Sequentially trains weak learners to minimize error, capturing complex patterns in data effectively.",
            "Linear Regression": "A simple baseline that models relationships linearly, offering high interpretability."
        }
