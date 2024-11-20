# app.py

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from utils.api_client import APIClient
from utils.predictor import Predictor
from datetime import datetime, timedelta
import os
import logging
from logging.handlers import RotatingFileHandler
import json
from flask_migrate import Migrate
import time
import threading
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(handler)


# Initialize Flask app
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'z$eJ#8dL4KpN6rV0yB2wX1cI3aH5fG7mS9tU!iQ@oW3nR1bY0kP'),
    SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///nba_predictions.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    TEMPLATES_AUTO_RELOAD=True
)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# Initialize API client and predictor
API_KEY = os.environ.get('NBA_API_KEY', '89ce3afd40msh6fe1b4a34da6f2ep1f2bcdjsn4a84afd6514c')
api_client = APIClient(API_KEY)
predictor = Predictor(api_client)

# Models
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    coins = db.Column(db.Integer, default=1000)
    boost_points = db.Column(db.Integer, default=0)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    game_id = db.Column(db.String(80), nullable=False)
    home_team = db.Column(db.String(80), nullable=False)
    away_team = db.Column(db.String(80), nullable=False)
    predicted_winner = db.Column(db.String(80), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    home_score_pred = db.Column(db.Integer)
    away_score_pred = db.Column(db.Integer)
    actual_winner = db.Column(db.String(80))
    home_score_actual = db.Column(db.Integer)
    away_score_actual = db.Column(db.Integer)
    coins_wagered = db.Column(db.Integer, default=0)
    coins_won = db.Column(db.Integer, default=0)
    boost_points_earned = db.Column(db.Integer, default=0)
    accuracy_score = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='pending')  # pending, correct, incorrect
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    game_date = db.Column(db.DateTime, nullable=False)

    def calculate_rewards(self):
        """Calculate rewards based on prediction accuracy"""
        try:
            if self.status != 'pending':
                return False

            # Get actual scores
            if not self.home_score_actual or not self.away_score_actual:
                return False

            # Determine actual winner
            actual_winner = 'Home' if self.home_score_actual > self.away_score_actual else 'Away'

            # Update prediction status
            self.status = 'correct' if self.predicted_winner == actual_winner else 'incorrect'

            # Calculate score accuracy
            home_diff = abs(self.home_score_pred - self.home_score_actual)
            away_diff = abs(self.away_score_pred - self.away_score_actual)
            self.accuracy_score = 1 - ((home_diff + away_diff) / 200)  # Scale to 0-1

            # Calculate rewards
            if self.status == 'correct':
                # Base reward is 2x wagered amount
                self.coins_won = self.coins_wagered * 2

                # Bonus for high confidence correct predictions
                if self.confidence == 'high':
                    self.coins_won = int(self.coins_won * 1.5)

                # Bonus for accurate score prediction
                if self.accuracy_score > 0.9:
                    self.coins_won += 50
                    self.boost_points_earned = 2
                elif self.accuracy_score > 0.8:
                    self.coins_won += 25
                    self.boost_points_earned = 1

                # Update user's coins and boost points
                user = User.query.get(self.user_id)
                user.coins += self.coins_won
                user.boost_points += self.boost_points_earned

            return True

        except Exception as e:
            logger.error(f"Error calculating rewards: {str(e)}")
            return False

# Login manager
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Template filters
@app.template_filter('datetime')
def format_datetime(value, format='%Y-%m-%d %H:%M'):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return value  # Fallback: return the original string if conversion fails
    return value.strftime(format)

# Routes
@app.route('/')
@login_required
def index():
    try:
        current_date = datetime.now()
        logger.info(f"Loading index page for user {current_user.username}")
        
        # Get live games
        live_games = api_client.get_live_games()
        live_predictions = {}
        
        # Generate predictions for live games
        for game in live_games:
            live_pred = predictor.predict_live_game(game)
            if live_pred:
                live_predictions[game['id']] = live_pred
        
        # Get scheduled games
        status = api_client.check_games_status()
        template_data = {
            'current_date': current_date,
            'authenticated': True,
            'user_coins': current_user.coins,
            'user_boost': current_user.boost_points,
            'error': None,
            'live_games': live_games,
            'live_predictions': live_predictions,
            'scheduled_games': [],
            'predictions': {},
            'show_tomorrow': False,
            'selected_date': current_date.strftime('%Y-%m-%d')
        }
        
        if status['show_tomorrow']:
            tomorrow = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
            scheduled_games = api_client.get_schedule_for_date(tomorrow)
            
            # Get predictions from database
            predictions = {}
            for game in scheduled_games:
                game_id = str(game['id'])
                
                existing_pred = Prediction.query.filter_by(
                    user_id=current_user.id,
                    game_id=game_id
                ).first()
                
                if existing_pred:
                    predictions[game_id] = existing_pred.to_dict()
                else:
                    prediction = predictor.predict_game(
                        game['teams']['home']['id'],
                        game['teams']['away']['id'],
                        game_id
                    )
                    if prediction:
                        predictions[game_id] = prediction
            
            template_data.update({
                'scheduled_games': scheduled_games,
                'predictions': predictions,
                'show_tomorrow': True,
                'selected_date': tomorrow
            })
        
        return render_template('index.html', **template_data)
        
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('index.html', 
            error='Error loading dashboard data',
            current_date=datetime.now(),
            authenticated=True
        )

@app.route('/prediction_history')
@login_required
def prediction_history():
    """View prediction history - alias for predictions route"""
    return redirect(url_for('predictions'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password')
            confirm_password = request.form.get('confirm_password')
            
            # Validation
            if not all([username, email, password, confirm_password]):
                flash('All fields are required', 'error')
                return redirect(url_for('register'))
                
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return redirect(url_for('register'))
                
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'error')
                return redirect(url_for('register'))
                
            if User.query.filter_by(email=email).first():
                flash('Email already registered', 'error')
                return redirect(url_for('register'))
            
            # Create new user
            user = User(username=username, email=email)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            db.session.rollback()
            flash('An error occurred during registration', 'error')
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            remember = request.form.get('remember', False)
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            else:
                flash('Invalid username or password', 'error')
                
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login', 'error')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/predictions')
@login_required
def predictions():
    try:
        # Get user's prediction history
        user_predictions = Prediction.query.filter_by(
            user_id=current_user.id
        ).order_by(Prediction.game_date.desc()).all()
        
        # Get upcoming games for next 7 days
        upcoming_games = []
        current_date = datetime.now()
        for i in range(7):
            date = (current_date + timedelta(days=i)).strftime('%Y-%m-%d')
            games = api_client.get_schedule_for_date(date)
            if games:
                for game in games:
                    # Check if prediction exists
                    prediction = Prediction.query.filter_by(
                        user_id=current_user.id,
                        game_id=str(game['id'])
                    ).first()
                    
                    upcoming_games.append({
                        'game': game,
                        'prediction': prediction.to_dict() if prediction else None,
                        'date': date
                    })
        
        return render_template(
            'predictions.html',
            predictions=user_predictions,
            upcoming_games=upcoming_games,
            total_predictions=len(user_predictions),
            current_user=current_user
        )
        
    except Exception as e:
        logger.error(f"Error in predictions route: {str(e)}")
        flash('Error loading predictions', 'error')
        return redirect(url_for('index'))

@app.route('/make_prediction', methods=['POST'])
@login_required
def make_prediction():
    try:
        data = request.get_json()
        game_id = data.get('game_id')
        
        # Check if prediction already exists
        existing_prediction = Prediction.query.filter_by(
            user_id=current_user.id,
            game_id=game_id
        ).first()
        
        if existing_prediction:
            return jsonify({
                'status': 'exists',
                'prediction': existing_prediction.to_dict()
            })
        
        # Get game details and generate prediction
        game = api_client.get_game_details(game_id)
        prediction = predictor.predict_game(
            game['teams']['home']['id'],
            game['teams']['away']['id']
        )
        
        # Save new prediction
        new_prediction = Prediction(
            user_id=current_user.id,
            game_id=game_id,
            home_team=game['teams']['home']['name'],
            away_team=game['teams']['away']['name'],
            predicted_winner=prediction['predicted_winner'],
            confidence=prediction['confidence_level'],
            home_score_pred=prediction['predicted_scores']['home']['mid'],
            away_score_pred=prediction['predicted_scores']['away']['mid'],
            game_date=datetime.strptime(game['date']['start'], '%Y-%m-%dT%H:%M:%SZ')
        )
        
        db.session.add(new_prediction)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'prediction': new_prediction.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/live_games')
@login_required
def live_games():
    try:
        games = api_client.get_live_games()
        predictions = {}
        
        for game in games:
            # Get existing prediction if any
            existing_pred = Prediction.query.filter_by(
                user_id=current_user.id,
                game_id=str(game['id'])
            ).first()
            
            if existing_pred:
                predictions[game['id']] = {
                    'predicted_winner': existing_pred.predicted_winner,
                    'confidence': existing_pred.confidence,
                    'home_score': existing_pred.home_score_pred,
                    'away_score': existing_pred.away_score_pred
                }
            else:
                # Generate new live prediction
                live_pred = predictor.predict_live_game(game)
                if live_pred:
                    predictions[game['id']] = live_pred
        
        return render_template(
            'live_games.html',
            games=games,
            predictions=predictions,
            current_date=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in live games route: {str(e)}")
        flash('Error loading live games', 'error')
        return redirect(url_for('index'))

@app.route('/scheduled_games')
@login_required
def scheduled_games():
    try:
        date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        games = api_client.get_schedule_for_date(date_str)
        
        # Get existing predictions
        existing_predictions = Prediction.query.filter(
            Prediction.user_id == current_user.id,
            Prediction.game_date >= datetime.strptime(date_str, '%Y-%m-%d'),
            Prediction.game_date < datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=1)
        ).all()
        
        predictions_dict = {str(p.game_id): p for p in existing_predictions}
        
        return render_template(
            'scheduled_games.html',
            games=games,
            predictions=predictions_dict,
            selected_date=date_str,
            current_date=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in scheduled games route: {str(e)}")
        flash('Error loading scheduled games', 'error')
        return redirect(url_for('index'))

@app.route('/update_predictions')
@login_required
def update_predictions():
    """Update prediction statuses and calculate rewards"""
    try:
        # Get pending predictions
        pending_predictions = Prediction.query.filter_by(
            user_id=current_user.id,
            status='pending'
        ).all()

        updated_count = 0
        for prediction in pending_predictions:
            # Get game result
            game_result = api_client.get_game_result(prediction.game_id)
            if not game_result or game_result['status'] != 'Final':
                continue

            # Update actual scores
            prediction.home_score_actual = game_result['scores']['home']['points']
            prediction.away_score_actual = game_result['scores']['away']['points']

            # Calculate rewards
            if prediction.calculate_rewards():
                updated_count += 1

        if updated_count > 0:
            db.session.commit()
            return jsonify({
                'success': True,
                'updated_count': updated_count,
                'new_coins': current_user.coins,
                'new_boost_points': current_user.boost_points
            })

        return jsonify({'success': True, 'updated_count': 0})

    except Exception as e:
        logger.error(f"Error updating predictions: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/game_details/<game_id>')
@login_required
def game_details(game_id):
    try:
        game = api_client.get_game_details(game_id)
        if not game:
            return jsonify({'error': 'Game not found'}), 404
            
        # Get prediction if exists
        prediction = Prediction.query.filter_by(
            user_id=current_user.id,
            game_id=str(game_id)
        ).first()
        
        # Get additional analysis
        analysis = predictor.generate_prediction_insights({
            'game': game,
            'prediction': prediction
        }) if prediction else None
        
        return jsonify({
            'game': game,
            'prediction': prediction.to_dict() if prediction else None,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error getting game details: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/live_updates')
@login_required
def live_updates():
    try:
        live_games = api_client.get_live_games()
        updates = []
        
        for game in live_games:
            prediction = Prediction.query.filter_by(
                user_id=current_user.id,
                game_id=str(game['id'])
            ).first()
            
            if prediction:
                live_pred = predictor.predict_live_game(game)
                updates.append({
                    'game_id': game['id'],
                    'score': {
                        'home': game['scores']['home']['points'],
                        'away': game['scores']['away']['points']
                    },
                    'prediction': live_pred,
                    'original_prediction': prediction.to_dict()
                })
        
        return jsonify({'updates': updates})
        
    except Exception as e:
        logger.error(f"Error getting live updates: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predictions/<game_id>')
@login_required
def get_prediction(game_id):
    """Get automated prediction for a specific game"""
    try:
        game = api_client.get_game_details(game_id)
        if not game:
            return jsonify({'error': 'Game not found'}), 404
            
        prediction = predictor.predict_game(
            game['teams']['home']['id'],
            game['teams']['away']['id']
        )
        
        if not prediction:
            return jsonify({'error': 'Could not generate prediction'}), 500
            
        return jsonify({
            'game': {
                'id': game_id,
                'home_team': game['teams']['home']['name'],
                'away_team': game['teams']['away']['name'],
                'date': game['date']['start']
            },
            'prediction': prediction
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/daily_predictions')
@login_required
def daily_predictions():
    """Get all predictions for today's games"""
    try:
        games = api_client.get_todays_schedule()
        predictions = {}
        
        for game in games:
            prediction = predictor.predict_game(
                game['teams']['home']['id'],
                game['teams']['away']['id']
            )
            if prediction:
                predictions[game['id']] = prediction
        
        return jsonify({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'predictions': predictions
        })
        
    except Exception as e:
        logger.error(f"Error getting daily predictions: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/live_predictions')
@login_required
def live_predictions():
    """Get predictions for all live games"""
    try:
        live_games = api_client.get_live_games()
        predictions = {}
        
        for game in live_games:
            try:
                if not api_client._validate_game_data(game):
                    logger.warning(f"Invalid game data structure for game {game.get('id')}")
                    continue
                    
                live_prediction = predictor.predict_live_game(game)
                if live_prediction:
                    predictions[game['id']] = live_prediction
                    
            except Exception as e:
                logger.error(f"Error processing prediction for game {game.get('id')}: {str(e)}")
                continue
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error getting live predictions: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@app.route('/stats')
@login_required
def prediction_stats():
    """View prediction accuracy statistics"""
    try:
        stats = predictor.get_prediction_metrics()
        return render_template('stats.html', stats=stats)
    except Exception as e:
        logger.error(f"Error getting prediction stats: {str(e)}")
        flash('Error loading prediction statistics', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

def init_app():
    """Initialize the application"""
    try:
        # Create database tables
        with app.app_context():
            db.create_all()
            
        # Set up logging
        if not app.debug:
            if not os.path.exists('logs'):
                os.mkdir('logs')
            file_handler = RotatingFileHandler(
                'logs/nba_predictions.log', 
                maxBytes=10240, 
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.INFO)
            app.logger.info('NBA Predictions startup')
            
        # Start cache cleanup task
        def cleanup_cache():
            while True:
                predictor.cache.clear_old_predictions()
                time.sleep(3600)  # Clean up every hour
        
        cleanup_thread = threading.Thread(target=cleanup_cache)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        
    except Exception as e:
        print(f"Error initializing application: {e}")
        raise

@app.cli.command("init-db")
def init_db_command():
    """Clear existing data and create new tables."""
    init_app()
    print("Initialized the database.")

@app.route('/api/scheduled-games')
@login_required
def api_scheduled_games():
    try:
        date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        games = api_client.get_schedule_for_date(date_str)
        return jsonify({'games': games})
    except Exception as e:
        logger.error(f"Error in scheduled games API: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/update_predictions', methods=['POST'])
@login_required
def update_prediction_status():
    """Update prediction statuses and calculate rewards"""
    try:
        # Get pending predictions
        pending_predictions = Prediction.query.filter_by(
            user_id=current_user.id,
            status='pending'
        ).all()

        updated_count = 0
        for prediction in pending_predictions:
            # Get actual game result
            game_result = api_client.get_game_result(prediction.game_id)
            if not game_result or game_result['status'] != 'Final':
                continue

            # Calculate rewards
            actual_score = {
                'home': game_result['scores']['home']['points'],
                'away': game_result['scores']['away']['points']
            }
            rewards = prediction.calculate_rewards(actual_score)

            if rewards:
                # Update user's coins and boost points
                if prediction.status == 'correct':
                    current_user.coins += rewards['coins_won']
                    current_user.boost_points += rewards['boost_points']
                updated_count += 1

        if updated_count > 0:
            db.session.commit()
            return jsonify({
                'updated': True,
                'count': updated_count,
                'new_coins': current_user.coins,
                'new_boost_points': current_user.boost_points
            })

        return jsonify({'updated': False})

    except Exception as e:
        logger.error(f"Error updating predictions: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/prediction_stats')
@login_required
def get_prediction_stats():
    """Get user's prediction statistics"""
    try:
        predictions = Prediction.query.filter_by(user_id=current_user.id).all()
        total = len(predictions)
        correct = len([p for p in predictions if p.status == 'correct'])
        
        return jsonify({
            'total_predictions': total,
            'correct_predictions': correct,
            'accuracy_rate': (correct / total * 100) if total > 0 else 0,
            'total_coins': current_user.coins,
            'boost_points': current_user.boost_points,
            'recent_predictions': [p.to_dict() for p in predictions[-5:]]
        })

    except Exception as e:
        logger.error(f"Error getting prediction stats: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/scheduled_games_players')
@login_required
def scheduled_games_players():
    try:
        logger.info("Starting scheduled games with players route")
        current_date = datetime.now()
        
        logger.info(f"Current date: {current_date}")
        games_with_details = []

        for i in range(3):
            date_str = (current_date + timedelta(days=i)).strftime('%Y-%m-%d')
            logger.info(f"\n{'='*50}\nFetching games for {date_str}\n{'='*50}")
            
            games = api_client.get_schedule_for_date(date_str)
            logger.info(f"\nScheduled Games Response Structure:")
            logger.info(f"Number of games: {len(games) if games else 0}")
            if games:
                logger.info(f"Sample game structure: {json.dumps(games[0], indent=2)}")
            
            # Process each game
            for game in games:
                logger.info(f"\n{'-'*50}\nProcessing Game ID: {game.get('id')}\n{'-'*50}")
                
                home_team_id = game['teams']['home']['id']
                away_team_id = game['teams']['away']['id']
                
                logger.info(f"\nTeam IDs:")
                logger.info(f"Home Team ID: {home_team_id} ({game['teams']['home']['name']})")
                logger.info(f"Away Team ID: {away_team_id} ({game['teams']['away']['name']})")
                
                # Get team details
                home_team_details = api_client.get_team_details(home_team_id)
                away_team_details = api_client.get_team_details(away_team_id)
                
                logger.info(f"\nTeam Details Response:")
                logger.info(f"Home Team Details: {json.dumps(home_team_details, indent=2)}")
                logger.info(f"Away Team Details: {json.dumps(away_team_details, indent=2)}")
                
                # Get rosters
                home_roster = api_client.get_team_roster(home_team_id)
                away_roster = api_client.get_team_roster(away_team_id)
                
                logger.info(f"\nRoster Information:")
                logger.info(f"Home Roster Size: {len(home_roster) if home_roster else 0}")
                if home_roster:
                    logger.info(f"Sample Home Player Structure: {json.dumps(home_roster[0], indent=2)}")
                logger.info(f"Away Roster Size: {len(away_roster) if away_roster else 0}")
                if away_roster:
                    logger.info(f"Sample Away Player Structure: {json.dumps(away_roster[0], indent=2)}")
                
                # Get recent performance
                home_recent_performance = api_client.get_team_recent_performance(home_team_id)
                away_recent_performance = api_client.get_team_recent_performance(away_team_id)
                
                logger.info(f"\nRecent Performance Data:")
                logger.info(f"Home Team Performance: {json.dumps(home_recent_performance, indent=2)}")
                logger.info(f"Away Team Performance: {json.dumps(away_recent_performance, indent=2)}")
                
                # Process top performers
                home_top_performers = [
                    player for player in (home_roster or [])
                    if player.get('is_top_performer', False)
                ]
                away_top_performers = [
                    player for player in (away_roster or [])
                    if player.get('is_top_performer', False)
                ]
                
                logger.info(f"\nTop Performers:")
                logger.info(f"Home Team Top Performers: {len(home_top_performers)}")
                logger.info(f"Away Team Top Performers: {len(away_top_performers)}")
                
                # Create game details
                game_details = {
                    'game_info': game,
                    'home_team': {
                        'name': game['teams']['home']['name'],
                        'team_details': home_team_details,
                        'roster': home_roster,
                        'recent_performance': home_recent_performance,
                        'top_performers': home_top_performers[:3]
                    },
                    'away_team': {
                        'name': game['teams']['away']['name'],
                        'team_details': away_team_details,
                        'roster': away_roster,
                        'recent_performance': away_recent_performance,
                        'top_performers': away_top_performers[:3]
                    },
                    'game_date': date_str
                }
                
                games_with_details.append(game_details)
                logger.info(f"\nGame Details Added:")
                logger.info(f"Game ID: {game['id']}")
                logger.info(f"Total Games Processed: {len(games_with_details)}")

        # Get existing predictions
        logger.info(f"\n{'='*50}\nFetching Predictions\n{'='*50}")
        existing_predictions = Prediction.query.filter(
            Prediction.user_id == current_user.id,
            Prediction.game_date >= current_date,
            Prediction.game_date < current_date + timedelta(days=3)
        ).all()
        
        predictions_dict = {str(p.game_id): p for p in existing_predictions}
        
        logger.info(f"\nPredictions Summary:")
        logger.info(f"Total Predictions: {len(predictions_dict)}")
        if predictions_dict:
            logger.info(f"Sample Prediction: {vars(list(predictions_dict.values())[0])}")
        
        logger.info(f"\n{'='*50}\nRendering Template\n{'='*50}")
        logger.info(f"Total Games to Display: {len(games_with_details)}")
        
        return render_template(
            'scheduled_games_players.html',
            games=games_with_details,
            predictions=predictions_dict,
            current_date=current_date
        )
        
    except Exception as e:
        logger.error(f"Error in scheduled games with players route: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        flash('Error loading scheduled games with players', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

