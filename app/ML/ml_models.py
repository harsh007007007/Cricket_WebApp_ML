import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import joblib

class CricketPerformanceModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.regressor = LinearRegression()
        self.similarity = NearestNeighbors(n_neighbors=6)
    
    def clean_data(self, df):
        """Convert all columns to numeric, handling dashes and other non-numeric values"""
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col].replace('-', np.nan), errors='coerce')
        return df.fillna(0)
    
    def prepare_features(self, df):
        """Feature engineering from raw data"""
        df = self.clean_data(df.copy())
        features = pd.DataFrame()
        
        # Batting features
        for fmt in ['Tests', 'ODIs', 'T20s']:
            features[f'bat_{fmt}_avg'] = df[f'BATTING_{fmt}_Ave']
            features[f'bat_{fmt}_sr'] = df[f'BATTING_{fmt}_SR']
        
        # Bowling features
        for fmt in ['Tests', 'ODIs', 'T20s']:
            features[f'bowl_{fmt}_avg'] = df[f'BOWLING_{fmt}_Ave']
            features[f'bowl_{fmt}_econ'] = df[f'BOWLING_{fmt}_Econ']
        
        return features

    def train_models(self, df):
        """Train all models with cleaned data"""
        features = self.prepare_features(df)
        X = self.scaler.fit_transform(features)
        
        # Train clustering
        self.kmeans.fit(X)
        
        # Train regression (predict Test runs)
        y = df['BATTING_Tests_Runs'].values
        self.regressor.fit(X, y)
        
        # Train similarity model
        self.similarity.fit(X)
        return self
    
    def predict_cluster(self, features):
        """Predict player cluster"""
        X = self.scaler.transform(features)
        return self.kmeans.predict(X)[0]
    
    def predict_performance(self, features):
        """Predict future performance"""
        X = self.scaler.transform(features)
        pred = self.regressor.predict(X)[0]
        return [pred, 0.8]  # Prediction and dummy confidence
    
    def get_top_performers(self, format_type, df):
        # Example logic – adjust column names as appropriate
        if format_type == 'test':
            # Sort by test runs (descending) to get top 10
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

        # Convert these DataFrame rows to dictionaries for the template
        top_batsmen = top_batsmen_df.to_dict('records')
        top_bowlers = top_bowlers_df.to_dict('records')

        return {
            'batsmen': top_batsmen,
            'bowlers': top_bowlers
        }