from flask import render_template, send_from_directory, jsonify, request
import pandas as pd
import json
import os
import random
import joblib

from app import app
from app.ML.ml_models import CricketPerformanceModel

# Base directory (one level above app/)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load datasets
df = pd.read_csv(os.path.join(base_dir, 'data', 'flattened_cricket_data.csv'), low_memory=False)
with open(os.path.join(base_dir, 'data', 'structured_cricket_data.json')) as f:
    players_json = json.load(f)

# Load or train model
model_path = os.path.join(base_dir, 'models', 'performance_model.pkl')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"[INFO] Model not found or error loading: {e}\n[INFO] Training new model...")
    model = CricketPerformanceModel()
    df_clean = model.clean_data(df.copy())
    model.train_models(df_clean)
    joblib.dump(model, model_path)
    print("[INFO] New model trained and saved.")

@app.route('/')
def home():
    return format_analysis('test')

@app.route('/analysis/<format_type>')
def format_analysis(format_type):
    if format_type not in ['test', 'odi', 't20']:
        return render_template('404.html'), 404

    viz_mapping = {
        'test': ['viz/top_test_batsmen.html', 'viz/top_test_bowlers.html', 'viz/test_allrounders.html'],
        'odi': ['viz/top_odi_batsmen.html', 'viz/odi_batting_impact.html', 'viz/top_odi_bowlers.html', 'viz/odi_allrounders.html', 'viz/odi_role_analysis.html'],
        't20': ['viz/top_t20_batsmen.html', 'viz/t20_batting_impact.html', 'viz/t20_strike_rates.html', 'viz/top_t20_bowlers.html', 'viz/t20_wicket_takers.html', 'viz/t20_allrounders.html']
    }

    existing_files = [
        viz_file.replace('viz/', '') for viz_file in viz_mapping[format_type]
        if os.path.exists(os.path.join(app.static_folder, viz_file))
    ]

    top_performers = model.get_top_performers(format_type, df)
    return render_template('analysis.html',
                           format=format_type,
                           viz_files=existing_files,
                           top_performers=top_performers)

@app.route('/players')
def player_list():
    # Filter to show only significant players, e.g., those with at least 10 matches.
    def filter_significant(players):
        significant = []
        for p in players:
            try:
                if (float(p.get('BATTING_Tests_Mat', 0)) >= 10 or
                    float(p.get('BATTING_ODIs_Mat', 0)) >= 10 or
                    float(p.get('BATTING_T20Is_Mat', 0)) >= 10):
                    significant.append(p)
            except Exception:
                continue
        return significant
    
    significant_players = filter_significant(players_json)
    random.shuffle(significant_players)
    selected_players = significant_players[:15]
    return render_template('players.html', players=selected_players)

@app.route('/player/<int:player_id>')
def player_profile(player_id):
    # Find the player by unique ID.
    player = next((p for p in players_json if int(p['ID']) == player_id), None)
    if not player:
        return render_template('404.html'), 404

    features = model.prepare_features(pd.DataFrame([player]))
    # Use clustering if available; otherwise, just set a default value.
    try:
        clusters = model.predict_cluster(features)
    except Exception:
        clusters = []
    
    # If find_similar_players is not implemented, return an empty list.
    if hasattr(model, 'find_similar_players'):
         similar_players = model.find_similar_players(features, players_json)
    else:
         similar_players = []
    
    return render_template('player.html',
                           player=player,
                           clusters=clusters,
                           similar_players=similar_players[:5])

@app.route('/api/search_players', methods=['GET'])
def search_players():
    query = request.args.get('q', '').lower().strip()
    results = []
    if query:
        for p in players_json:
            # Get the player name safely; convert non-string values to string
            name_val = p.get('NAME', '')
            if name_val is None:
                continue
            name_str = str(name_val).strip()  # Ensure it's a string
            if query in name_str.lower():
                results.append({
                    'ID': p.get('ID', ''),
                    'NAME': name_str,
                    'COUNTRY': p.get('COUNTRY', '')
                })
    return jsonify(results[:10])

@app.route('/api/player/<int:player_id>/predict')
def predict_performance(player_id):
    player = next((p for p in players_json if int(p['ID']) == player_id), None)
    if not player:
        return jsonify({'error': 'Player not found'}), 404

    features = model.prepare_features(pd.DataFrame([player]))
    prediction = model.predict_performance(features)
    return jsonify({
        'name': player['NAME'],
        'predicted_runs': prediction[0],
        'confidence': prediction[1]
    })

@app.route('/viz/<path:filename>')
def viz_files(filename):
    return send_from_directory(os.path.join(app.static_folder, 'viz'), filename)
