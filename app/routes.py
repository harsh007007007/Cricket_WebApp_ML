from flask import render_template
from app import app

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/players')
def players():
    return render_template('players.html')

@app.route('/teams')
def teams():
    return render_template('teams.html')

@app.route('/visualizations')
def visualizations():
    return render_template('visualizations.html')

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')