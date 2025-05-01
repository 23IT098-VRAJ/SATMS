import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import threading
import schedule
import logging
from functools import wraps
import zipfile
import shutil
import pickle
from flask_socketio import SocketIO, emit
import random
from scipy import stats
import math
from src.traffic_predictor import TrafficPredictor

# Flask and authentication imports
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import re

# Create necessary directories
os.makedirs('logs', exist_ok=True)
os.makedirs('config', exist_ok=True)
os.makedirs('config/ssl', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('backups', exist_ok=True)
os.makedirs('models/saved_models', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Backup System class
class BackupSystem:
    """System for backing up and restoring application data"""
    
    def __init__(self, config_dir='config', data_dir='data', backup_dir='backups'):
        """Initialize the backup system
        
        Args:
            config_dir: Directory containing configuration files
            data_dir: Directory containing data files
            backup_dir: Directory to store backups
        """
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"Backup system initialized with backup directory: {backup_dir}")
    
    def create_backup(self):
        """Create a backup of configuration and data
        
        Returns:
            str: Path to the created backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{self.backup_dir}/backup_{timestamp}.zip"
        
        with zipfile.ZipFile(backup_filename, 'w') as backup_zip:
            # Backup configuration files
            if os.path.exists(self.config_dir):
                for root, _, files in os.walk(self.config_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path)
                        backup_zip.write(file_path, arc_name)
            
            # Backup data files
            if os.path.exists(self.data_dir):
                for root, _, files in os.walk(self.data_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path)
                        backup_zip.write(file_path, arc_name)
        
        logger.info(f"Backup created: {backup_filename}")
        return backup_filename
    
    def restore_backup(self, backup_file):
        """Restore from a backup file
        
        Args:
            backup_file: Path to the backup file
            
        Returns:
            bool: True if restoration was successful
        """
        if not os.path.exists(backup_file):
            logger.error(f"Backup file {backup_file} not found")
            return False
            
        # Create temporary restore directory
        restore_dir = f"{self.backup_dir}/restore_temp"
        if os.path.exists(restore_dir):
            shutil.rmtree(restore_dir)
        os.makedirs(restore_dir, exist_ok=True)
        
        # Extract backup
        with zipfile.ZipFile(backup_file, 'r') as backup_zip:
            backup_zip.extractall(restore_dir)
        
        # Restore configuration
        if os.path.exists(f"{restore_dir}/{self.config_dir}"):
            if os.path.exists(self.config_dir):
                shutil.rmtree(self.config_dir)
            shutil.copytree(f"{restore_dir}/{self.config_dir}", self.config_dir)
            
        # Restore data
        if os.path.exists(f"{restore_dir}/{self.data_dir}"):
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
            shutil.copytree(f"{restore_dir}/{self.data_dir}", self.data_dir)
            
        # Clean up
        shutil.rmtree(restore_dir)
        
        logger.info(f"Restored from backup: {backup_file}")
        return True

# Create user database manager
class UserDatabase:
    """Manages persistent user storage"""
    
    def __init__(self, db_file='data/users.db'):
        """Initialize the user database
        
        Args:
            db_file: Path to the database file
        """
        self.db_file = db_file
        self.users = {}
        self.load()
    
    def load(self):
        """Load users from file"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'rb') as f:
                    self.users = pickle.load(f)
                logger.info(f"Loaded {len(self.users)} users from database")
            else:
                logger.info("No existing user database found, creating new")
                self.users = {}
                # Create default admin user with correct credentials
                self.users['admin@example.com'] = {
                    'password': generate_password_hash('Admin123'),
                    'role': 'admin'
                }
                self.save()
        except Exception as e:
            logger.error(f"Error loading user database: {e}")
            # Create default admin user if there's an error
            self.users = {
                'admin@example.com': {
                    'password': generate_password_hash('Admin123'),
                    'role': 'admin'
                }
            }
            self.save()
    
    def save(self):
        """Save users to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
            
            with open(self.db_file, 'wb') as f:
                pickle.dump(self.users, f)
            logger.info(f"Saved {len(self.users)} users to database")
        except Exception as e:
            logger.error(f"Error saving user database: {e}")
    
    def add_user(self, email, password_hash, role='user'):
        """Add a new user
        
        Args:
            email: User email
            password_hash: Hashed password
            role: User role (default: user)
            
        Returns:
            bool: True if successful
        """
        if email in self.users:
            return False
        
        self.users[email] = {
            'password': password_hash,
            'role': role
        }
        self.save()
        return True
    
    def get_user(self, email):
        """Get user by email
        
        Args:
            email: User email
            
        Returns:
            dict: User data or None
        """
        return self.users.get(email)
    
    def verify_password(self, email, password):
        """Verify user password
        
        Args:
            email: User email
            password: Plain password to check
            
        Returns:
            bool: True if password matches
        """
        user = self.get_user(email)
        if not user:
            return False
            
        try:
            return check_password_hash(user['password'], password)
        except Exception as e:
            logger.error(f"Error verifying password: {e}")
            return False
    
    def is_admin(self, email):
        """Check if user is an admin
        
        Args:
            email: User email
            
        Returns:
            bool: True if user is an admin
        """
        user = self.get_user(email)
        if not user:
            return False
        
        return user.get('role') == 'admin'

# Create input validation system
class InputValidator:
    """System for validating user inputs"""
    
    @staticmethod
    def validate_email(email):
        """Validate email format
        
        Args:
            email: Email address to validate
            
        Returns:
            bool: True if valid
        """
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_password(password):
        """Validate password strength
        
        Args:
            password: Password to validate
            
        Returns:
            bool: True if valid
        """
        # At least 8 characters, with uppercase, lowercase, and number
        pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$'
        return bool(re.match(pattern, password))
    
    @staticmethod
    def sanitize_input(input_str):
        """Sanitize input to prevent XSS
        
        Args:
            input_str: Input string to sanitize
            
        Returns:
            str: Sanitized string
        """
        if not isinstance(input_str, str):
            return input_str
            
        # Replace potentially dangerous characters
        sanitized = input_str.replace('<', '&lt;').replace('>', '&gt;')
        return sanitized

# Create Flask app with proper template and static folders
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for API endpoints
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize user database
user_db = UserDatabase()

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login_page'))
        if not user_db.is_admin(session['user']):
            return render_template('error.html', error="Access Denied", 
                                 message="You do not have permission to access this area.")
        return f(*args, **kwargs)
    return decorated_function

# Flask routes
@app.route('/')
def index():
    """Home page route"""
    if 'user' in session:
        is_admin = user_db.is_admin(session['user'])
        return render_template('index.html', user=session['user'], is_admin=is_admin)
    else:
        return redirect(url_for('login_page'))

@app.route('/login', methods=['GET'])
def login_page():
    """Login page route"""
    return render_template('login.html')

@app.route('/register', methods=['GET'])
def register_page():
    """Registration page route"""
    return render_template('register.html')

@app.route('/admin/backup', methods=['GET'])
@admin_required
def admin_backup_page():
    """Admin backup management page route"""
    # List existing backups
    backup_dir = 'backups'
    backup_files = []
    if os.path.exists(backup_dir):
        backup_files = [f for f in os.listdir(backup_dir) if f.startswith('backup_') and f.endswith('.zip')]
        backup_files.sort(reverse=True)
    
    return render_template('admin/backup.html', backup_files=backup_files)

@app.route('/logout')
def logout():
    """Logout route"""
    session.pop('user', None)
    return redirect(url_for('login_page'))

@app.route('/dashboard')
@login_required
def dashboard_web():
    """Web dashboard route"""
    traffic_data = traffic_system.update_traffic_data()
    
    # Calculate average density for traffic flow chart
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Prepare traffic flow chart data with separate lines for each direction
    traffic_chart_data = {
        'labels': [current_time],
        'datasets': [
            {
                'label': 'North Traffic',
                'data': [traffic_data['densities']['north'] * 100],
                'borderColor': '#2196F3',
                'backgroundColor': 'rgba(33, 150, 243, 0.1)',
                'tension': 0.4,
                'fill': True
            },
            {
                'label': 'South Traffic',
                'data': [traffic_data['densities']['south'] * 100],
                'borderColor': '#4CAF50',
                'backgroundColor': 'rgba(76, 175, 80, 0.1)',
                'tension': 0.4,
                'fill': True
            },
            {
                'label': 'East Traffic',
                'data': [traffic_data['densities']['east'] * 100],
                'borderColor': '#FFC107',
                'backgroundColor': 'rgba(255, 193, 7, 0.1)',
                'tension': 0.4,
                'fill': True
            },
            {
                'label': 'West Traffic',
                'data': [traffic_data['densities']['west'] * 100],
                'borderColor': '#FF5722',
                'backgroundColor': 'rgba(255, 87, 34, 0.1)',
                'tension': 0.4,
                'fill': True
            }
        ]
    }
    
    # Prepare density chart data
    density_chart_data = {
        'labels': ['North', 'South', 'East', 'West'],
        'datasets': [{
            'label': 'Current Density',
            'data': [
                traffic_data['densities']['north'] * 100,
                traffic_data['densities']['south'] * 100,
                traffic_data['densities']['east'] * 100,
                traffic_data['densities']['west'] * 100
            ],
            'backgroundColor': [
                'rgba(33, 150, 243, 0.8)',
                'rgba(76, 175, 80, 0.8)',
                'rgba(255, 193, 7, 0.8)',
                'rgba(255, 87, 34, 0.8)'
            ],
            'borderColor': [
                'rgba(33, 150, 243, 1)',
                'rgba(76, 175, 80, 1)',
                'rgba(255, 193, 7, 1)',
                'rgba(255, 87, 34, 1)'
            ],
            'borderWidth': 2,
            'borderRadius': 5
        }]
    }
    
    return render_template('dashboard.html',
                         traffic_data=traffic_data,
                         traffic_chart_data=traffic_chart_data,
                         density_chart_data=density_chart_data)

# API routes
@app.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint for login"""
    data = request.json
    email = InputValidator.sanitize_input(data.get('email', ''))
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    if user_db.verify_password(email, password):
        session['user'] = email
        return jsonify({'message': 'Login successful'})
    
    return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/api/register', methods=['POST'])
def api_register():
    """API endpoint for registration"""
    data = request.json
    email = InputValidator.sanitize_input(data.get('email', ''))
    password = data.get('password', '')
    
    # Validate inputs
    if not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    if not InputValidator.validate_email(email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    if not InputValidator.validate_password(password):
        return jsonify({'error': 'Password must be at least 8 characters with uppercase, lowercase, and number'}), 400
    
    if user_db.get_user(email):
        return jsonify({'error': 'Email already registered'}), 400
    
    # Create user
    password_hash = generate_password_hash(password)
    if user_db.add_user(email, password_hash):
        return jsonify({
            'message': 'Registration successful! You can now log in.',
            'email': email
        })
    else:
        return jsonify({'error': 'Failed to register user'}), 500

@app.route('/api/backup/create', methods=['POST'])
@admin_required
def api_backup_create():
    """API endpoint to create a backup"""
    backup_system = BackupSystem()
    try:
        backup_file = backup_system.create_backup()
        return jsonify({
            'message': f'Backup created successfully: {os.path.basename(backup_file)}',
            'file': os.path.basename(backup_file)
        })
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")
        return jsonify({'error': 'Failed to create backup'}), 500

@app.route('/api/backup/restore', methods=['POST'])
@admin_required
def api_backup_restore():
    """API endpoint to restore from a backup"""
    data = request.json
    file = InputValidator.sanitize_input(data.get('file', ''))
    
    if not file or not file.startswith('backup_') or not file.endswith('.zip'):
        return jsonify({'error': 'Invalid backup file'}), 400
    
    backup_system = BackupSystem()
    backup_path = os.path.join('backups', file)
    
    try:
        if backup_system.restore_backup(backup_path):
            return jsonify({'message': 'Backup restored successfully'})
        else:
            return jsonify({'error': 'Failed to restore backup'}), 500
    except Exception as e:
        logger.error(f"Backup restoration failed: {e}")
        return jsonify({'error': f'Failed to restore backup: {str(e)}'}), 500

@app.route('/api/backup/delete', methods=['POST'])
@admin_required
def api_backup_delete():
    """API endpoint to delete a backup"""
    data = request.json
    file = InputValidator.sanitize_input(data.get('file', ''))
    
    if not file or not file.startswith('backup_') or not file.endswith('.zip'):
        return jsonify({'error': 'Invalid backup file'}), 400
    
    backup_path = os.path.join('backups', file)
    
    try:
        if os.path.exists(backup_path):
            os.remove(backup_path)
            return jsonify({'message': 'Backup deleted successfully'})
        else:
            return jsonify({'error': 'Backup file not found'}), 404
    except Exception as e:
        logger.error(f"Backup deletion failed: {e}")
        return jsonify({'error': f'Failed to delete backup: {str(e)}'}), 500

def scheduled_backup():
    """Create a scheduled backup"""
    backup_system = BackupSystem()
    backup_file = backup_system.create_backup()
    logger.info(f"Scheduled backup created: {backup_file}")

def run_scheduler():
    """Run the scheduler for automated tasks"""
    # Schedule daily backup at midnight
    schedule.every().day.at("00:00").do(scheduled_backup)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# Initialize SocketIO with async mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    # Send initial data
    emit_dashboard_update()

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('request_update')
def handle_update_request():
    emit_dashboard_update()

def emit_dashboard_update():
    """Emit dashboard update via WebSocket"""
    try:
        traffic_data = traffic_system.update_traffic_data()
        current_time = datetime.now().strftime('%H:%M:%S')
        
        # Calculate average density
        avg_density = sum(traffic_data['densities'].values()) / len(traffic_data['densities']) * 100
        
        # Prepare data for WebSocket emission
        socketio.emit('dashboard_update', {
            'stats': {
                'vehicles': traffic_data['total_vehicles'],
                'density': f"{avg_density:.1f}%",
                'signals': traffic_data['active_green'],  # Use the active_green count directly
                'emergency': traffic_data['emergency_count']
            },
            'densities': traffic_data['densities'],
            'predictions': traffic_data['predictions'],
            'states': traffic_data['states'],
            'wait_times': traffic_data['wait_times'],
            'elapsed_waits': traffic_data['elapsed_waits'],
            'wait_percentages': traffic_data['wait_percentages'],
            'events': traffic_data['events']
        })
        
        # Log more detailed debug information
        logger.debug(f"Dashboard update: vehicles={traffic_data['total_vehicles']}, " +
                  f"density={avg_density:.1f}%, green={traffic_data['active_green']}, " +
                  f"phase={traffic_data['phase_state']}, wait_times={traffic_data['wait_times']}")
        
    except Exception as e:
        logger.error(f"Error in dashboard update: {e}")
        import traceback
        logger.error(traceback.format_exc())

def update_dashboard():
    """Background task to update dashboard periodically"""
    while True:
        try:
            emit_dashboard_update()
        except Exception as e:
            logger.error(f"Error in periodic dashboard update: {e}")
            import traceback
            logger.error(traceback.format_exc())
        time.sleep(5)  # Update every 5 seconds instead of every second

class TrafficLight:
    def __init__(self, intersection_id, location):
        self.intersection_id = intersection_id
        self.location = location
        self.state = "red"
        self.wait_time = 0
        self.vehicle_count = 0
        self.last_update = time.time()
        self.history = []  # Store historical data for prediction
        self.prediction_window = 5  # Number of time steps to predict
        self.min_green_time = 15  # Minimum green time in seconds
        self.max_green_time = 40  # Maximum green time in seconds (reduced for more frequent changes)
        self.emergency_priority = False
        self.time_in_state = 0  # How long the light has been in current state
        
    def update_state(self, traffic_density, predicted_density=None, recommended_green_time=None):
        """Update traffic light state based on current and predicted traffic"""
        current_time = time.time()
        time_diff = current_time - self.last_update
        
        # Update time in current state
        self.time_in_state += time_diff
        
        # Update history
        self.history.append(traffic_density)
        if len(self.history) > 10:  # Keep last 10 readings
            self.history.pop(0)
            
        # Update wait time if red
        if self.state == "red":
            self.wait_time += time_diff
        else:
            self.wait_time = 0
            
        # Use recommended green time if provided, otherwise calculate
        optimal_green_time = recommended_green_time if recommended_green_time else self.calculate_optimal_green_time(traffic_density)
        
        # State machine for traffic light
        # If state is forced by parent system, we don't change it here
        previous_state = self.state
        
        # Emergency priority handling
        if self.emergency_priority:
            self.state = "green"
            self.emergency_priority = False
            self.time_in_state = 0
        # Normal traffic flow handling
        elif self.state == "green" and self.time_in_state >= optimal_green_time:
            # Green has been active long enough, change to yellow
            self.state = "yellow"
            self.time_in_state = 0
        elif self.state == "yellow" and self.time_in_state >= 3:  # Fixed 3-second yellow
            # Yellow time finished, change to red
            self.state = "red"
            self.time_in_state = 0
        elif self.state == "red":
            # If red light has been active long enough and high traffic, change to green
            high_traffic = traffic_density > 0.6 or (predicted_density and predicted_density > 0.7)
            medium_traffic = traffic_density > 0.3 or (predicted_density and predicted_density > 0.4)
            
            if high_traffic and self.wait_time > optimal_green_time * 0.5:
                self.state = "green"
                self.time_in_state = 0
            elif medium_traffic and self.wait_time > optimal_green_time * 0.8:
                self.state = "green"
                self.time_in_state = 0
            elif self.wait_time > optimal_green_time * 1.2:
                # Prevent excessively long red lights regardless of traffic
                self.state = "green"
                self.time_in_state = 0
        
        # If state changed, reset timer
        if previous_state != self.state:
            self.time_in_state = 0
            
        self.last_update = current_time
        return self.state
        
    def calculate_optimal_green_time(self, traffic_density):
        """Calculate optimal green time based on traffic density"""
        # Linear interpolation between min and max time based on density
        return self.min_green_time + (self.max_green_time - self.min_green_time) * traffic_density
        
    def set_emergency_priority(self):
        """Set emergency priority for this traffic light"""
        self.emergency_priority = True
        # Reset time in state to trigger immediate change
        self.time_in_state = 0

class TrafficPredictor:
    def __init__(self):
        self.history = {}
        self.min_green_time = 15  # Minimum green time in seconds
        self.max_green_time = 60  # Maximum green time in seconds
    
    def update_history(self, densities):
        self.history = densities
    
    def predict_next_interval(self):
        predictions = {}
        for direction, density in self.history.items():
            predictions[direction] = self.predict_traffic(density)
        return predictions
    
    def predict_traffic(self, density):
        # Simple prediction with some randomness
        prediction = density * (1 + random.uniform(-0.1, 0.2))
        return max(0.1, min(1.0, prediction))
    
    def get_congestion_probability(self, direction, prediction):
        """Calculate probability of congestion based on current and predicted traffic
        
        Args:
            direction: Traffic direction
            prediction: Predicted traffic density
            
        Returns:
            float: Probability of congestion (0-1)
        """
        try:
            # Get current density
            current_density = self.history.get(direction, 0)
            
            # Calculate trend (if current density is higher than prediction)
            trend_factor = 1.0
            if current_density > prediction:
                trend_factor = 1.2  # Increasing congestion
            elif current_density < prediction * 0.5:
                trend_factor = 0.8  # Decreasing congestion
            
            # Calculate congestion probability
            congestion_prob = prediction * trend_factor
            
            # Ensure probability is within bounds
            return max(0.0, min(1.0, congestion_prob))
            
        except Exception as e:
            logger.error(f"Error calculating congestion probability: {e}")
            return prediction  # Fallback to raw prediction
    
    def get_recommended_green_time(self, direction, prediction):
        """Calculate recommended green light duration based on prediction and congestion
        
        Args:
            direction: Traffic direction
            prediction: Predicted traffic density
            
        Returns:
            int: Recommended green light duration in seconds
        """
        try:
            # Calculate congestion probability
            congestion_prob = self.get_congestion_probability(direction, prediction)
            
            # Base time calculation using congestion probability
            base_time = self.min_green_time + (self.max_green_time - self.min_green_time) * congestion_prob
            
            # Add variation based on current vs predicted density
            current_density = self.history.get(direction, 0)
            if current_density > prediction:
                # If current density is higher than predicted, increase time
                base_time *= 1.2
            elif current_density < prediction * 0.5:
                # If current density is much lower than predicted, decrease time
                base_time *= 0.8
            
            # Ensure time is within bounds
            return int(min(self.max_green_time, max(self.min_green_time, base_time)))
            
        except Exception as e:
            logger.error(f"Error calculating recommended green time: {e}")
            return 30  # Return default time on error

class TrafficManagementSystem:
    def __init__(self):
        self.intersections = {
            'north': TrafficLight('N1', 'North Junction'),
            'south': TrafficLight('S1', 'South Junction'),
            'east': TrafficLight('E1', 'East Junction'),
            'west': TrafficLight('W1', 'West Junction')
        }
        self.vehicle_data = []
        self.emergency_vehicles = set()
        self.last_update = time.time()
        self.total_vehicles = 30
        self.max_vehicles_per_hour = 100
        self.hour_stats = {hour: 0 for hour in range(24)}
        self.events = []
        self.last_vehicle_change = time.time()
        self.forced_reset_counter = 0
        
        # Wait time prediction
        self.initial_wait_times = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        self.predicted_wait_times = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
        
        # Initialize traffic predictor
        self.predictor = TrafficPredictor()
        
        # Traffic light phases
        self.phases = [
            {'name': 'NS', 'green': ['north', 'south'], 'yellow': [], 'red': ['east', 'west']},
            {'name': 'EW', 'green': ['east', 'west'], 'yellow': [], 'red': ['north', 'south']}
        ]
        self.current_phase_index = 0
        self.phase_start_time = time.time()
        self.min_phase_time = 15
        self.max_phase_time = 45
        self.yellow_time = 5  # Changed to 5 seconds as suggested
        self.wait_times = {direction: 0 for direction in self.intersections}
        self.phase_state = "GREEN"
        
        # Ensure system starts in a valid state
        self._initialize_lights()
        
        # Start with first phase active
        logger.info("Traffic Management System initialized with NS phase active")
        
    def _initialize_lights(self):
        """Initialize traffic lights to match the starting phase"""
        active_phase = self.phases[self.current_phase_index]
        
        # Ensure at least one direction is green
        if not active_phase['green']:
            logger.error("No green lights in active phase - forcing first phase")
            self.current_phase_index = 0
            active_phase = self.phases[self.current_phase_index]
        
        # Set initial states
        for direction in self.intersections:
            if direction in active_phase['green']:
                self.intersections[direction].state = 'green'
                self.wait_times[direction] = 0
                self.predicted_wait_times[direction] = 0
            else:
                self.intersections[direction].state = 'red'
                # Set initial wait times for red lights based on default phase time
                self.wait_times[direction] = 0
                self.predicted_wait_times[direction] = self.min_phase_time + self.yellow_time
                self.initial_wait_times[direction] = self.predicted_wait_times[direction]
                
        # Log initial state
        green_dirs = ", ".join(active_phase['green'])
        logger.info(f"Initialized traffic lights with {green_dirs} set to green")
        
    def _calculate_phase_time(self, densities, predictions, active_directions):
        """Calculate appropriate phase time based on traffic density"""
        if not active_directions:
            return self.min_phase_time
            
        # Start with minimum time
        base_time = self.min_phase_time
        
        # Get maximum density and prediction for active directions
        max_density = max(densities[d] for d in active_directions)
        
        # Calculate dynamic time based on density (1.0 to 3.0 multiplier)
        density_factor = 1.0 + max_density * 2
        
        # Calculate time
        phase_time = base_time * density_factor
        
        # Ensure within bounds
        return min(max(phase_time, self.min_phase_time), self.max_phase_time)
        
    def _calculate_wait_time(self, densities, green_directions):
        """Calculate predicted wait time for red directions based on green traffic"""
        # Get average density of green directions
        if not green_directions:
            return self.min_phase_time
            
        avg_green_density = sum(densities[d] for d in green_directions) / len(green_directions)
        
        # Calculate wait time - higher density in active directions = longer wait
        base_wait = self.min_phase_time
        density_factor = 1.0 + avg_green_density * 2.5  # 1.0 to 3.5 multiplier
        
        wait_time = base_wait * density_factor + self.yellow_time
        
        # Ensure within reasonable bounds
        return min(max(wait_time, self.min_phase_time), 90)  # Cap at 90 seconds
        
    def calculate_traffic_density(self, direction, current_hour):
        """Calculate realistic traffic density based on direction and time for a single intersection"""
        # Base flow rate for this hour
        base_rate = self._get_base_flow_rate(current_hour)
        
        # Direction-specific factors - more pronounced for a single intersection
        direction_factors = {
            'north': 1.2,  # Residential area with school/hospital (higher morning traffic)
            'south': 1.5,  # Commercial area (highest traffic)
            'east': 0.8,   # Residential area
            'west': 1.0    # Mixed use area
        }
        
        # Time-specific factors by direction (to create more realistic flow patterns)
        time_factors = {
            'north': 1.0 if 7 <= current_hour <= 9 else 0.7,  # Morning commute from residential
            'south': 1.2 if 16 <= current_hour <= 19 else 0.9,  # Evening shopping/dining
            'east': 0.9 if 11 <= current_hour <= 14 else 0.6,  # Lunch traffic
            'west': 1.1 if 16 <= current_hour <= 18 else 0.8   # Evening commute
        }
        
        # Calculate density with directional and time factors
        base_density = (base_rate / self.max_vehicles_per_hour) * direction_factors[direction] * time_factors[direction]
        
        # Add more random variation (±25%)
        variation = random.uniform(-0.25, 0.25)
        density = base_density + variation
        
        # Add larger periodic variation to simulate traffic waves
        time_factor = time.time() / 300  # 5-minute cycle
        density += 0.2 * math.sin(time_factor * math.pi * 2)
        
        # Add a second, faster oscillation for more realistic patterns
        fast_factor = time.time() / 60  # 1-minute cycle
        density += 0.1 * math.sin(fast_factor * math.pi * 2 + (hash(direction) % 10))
        
        # Ensure density is within limits (higher minimum for better visualization)
        return max(0.15, min(0.95, density))
        
    def update_traffic_data(self):
        """Update traffic data and manage traffic lights"""
        current_time = time.time()
        time_diff = current_time - self.last_update
        current_hour = datetime.now().hour
        
        # TRAFFIC FLOW CALCULATIONS
        # Calculate realistic vehicle flow with more variation
        base_flow = self._get_base_flow_rate(current_hour)
        
        # More dramatic flow variation (±50%)
        flow_variation = random.uniform(-0.5, 0.5)
        current_flow = base_flow * (1 + flow_variation)
        
        # Calculate new vehicles
        new_vehicles = int(current_flow * time_diff / 3600)
        
        # Add more complex periodicity to vehicle counts
        # Primary 10-minute cycle
        time_factor_1 = time.time() / 600
        sine_variation_1 = math.sin(time_factor_1 * 2 * math.pi) * 20
        
        # Secondary 3-minute cycle
        time_factor_2 = time.time() / 180
        sine_variation_2 = math.sin(time_factor_2 * 2 * math.pi) * 10
        
        # Combine variations
        sine_variation = sine_variation_1 + sine_variation_2
        
        # Ensure vehicle count changes every update
        time_since_change = current_time - self.last_vehicle_change
        
        if time_since_change > 5:  # Force change every 5 seconds at minimum
            # Add random jitter to ensure change
            jitter = random.randint(-5, 5)
            self.last_vehicle_change = current_time
        else:
            jitter = random.randint(-2, 2)  # Smaller changes during normal operation
        
        # Calculate new total with more realistic range (40-200 vehicles)
        new_total = max(40, min(200, int(self.total_vehicles + new_vehicles + sine_variation + jitter)))
        
        # Ensure the value actually changes
        if new_total == self.total_vehicles:
            new_total += random.choice([-1, 1]) * random.randint(1, 5)
            new_total = max(40, min(200, new_total))
            
        self.total_vehicles = new_total
        self.hour_stats[current_hour] += new_vehicles
        
        # TRAFFIC LIGHT MANAGEMENT
        # Calculate densities for each direction
        densities = {}
        for direction in self.intersections:
            density = self.calculate_traffic_density(direction, current_hour)
            densities[direction] = density
        
        # Update predictor with current densities
        self.predictor.update_history(densities)
        predictions = self.predictor.predict_next_interval()
        
        # Get the current phase configuration
        active_phase = self.phases[self.current_phase_index]
        phase_duration = current_time - self.phase_start_time
        
        # Track current states
        light_states = {}
        for direction, light in self.intersections.items():
            light_states[direction] = light.state
        
        # CRITICAL SAFETY CHECK: Immediately fix all-red state
        if all(state == 'red' for state in light_states.values()):
            self.forced_reset_counter += 1
            logger.error(f"CRITICAL ERROR: All lights red state detected ({self.forced_reset_counter} occurrences)")
            
            # Force system to a known good state
            self.current_phase_index = 0
            self.phase_state = "GREEN"
            self.phase_start_time = current_time
            
            # Immediately set N/S to green
            for direction in ['north', 'south']:
                self.intersections[direction].state = 'green'
                light_states[direction] = 'green'
                self.wait_times[direction] = 0
                self.predicted_wait_times[direction] = 0
            
            # Set E/W to red with predicted wait times
            predicted_wait = self._calculate_wait_time(densities, ['north', 'south'])
            for direction in ['east', 'west']:
                self.intersections[direction].state = 'red'
                light_states[direction] = 'red'
                self.wait_times[direction] = 0  # Reset actual wait time
                self.predicted_wait_times[direction] = predicted_wait
                self.initial_wait_times[direction] = predicted_wait
                
            # Update phase structure
            self.phases[0]['green'] = ['north', 'south']
            self.phases[0]['yellow'] = []
            self.phases[0]['red'] = ['east', 'west']
            
            # Record this emergency event
            self.events.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'type': 'critical',
                'message': f"EMERGENCY OVERRIDE: Fixed all-red state by forcing N/S to green"
            })
        
        # NORMAL PHASE MANAGEMENT (if we're not in a critical state)
        else:
            # STATE MACHINE LOGIC
            if self.phase_state == "GREEN":
                # We're in a stable green phase
                # Calculate how long this green phase should last based on traffic
                recommended_time = self._calculate_phase_time(
                    densities, 
                    predictions,
                    active_phase['green']
                )
                
                # Check if it's time to change to yellow (and we have green lights to change)
                if phase_duration >= recommended_time and active_phase['green']:
                    logger.info(f"Changing {active_phase['name']} from GREEN to YELLOW after {phase_duration:.1f}s")
                    
                    # Move green lights to yellow
                    active_phase['yellow'] = active_phase['green'].copy()
                    active_phase['green'] = []
                    
                    # Update light states
                    for direction in active_phase['yellow']:
                        self.intersections[direction].state = 'yellow'
                        light_states[direction] = 'yellow'
                    
                    # Change phase state
                    self.phase_state = "YELLOW"
                    self.phase_start_time = current_time
                    
                    # Log the change
                    self.events.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'type': 'phase_change',
                        'message': f"Changing {active_phase['name']} to YELLOW after {recommended_time:.1f}s"
                    })
            
            elif self.phase_state == "YELLOW":
                # We're in a yellow phase - this has fixed duration
                if phase_duration >= self.yellow_time:
                    logger.info(f"Yellow phase complete, transitioning to next phase")
                    
                    # Move to next phase
                    next_phase_index = (self.current_phase_index + 1) % len(self.phases)
                    next_phase = self.phases[next_phase_index]
                    
                    # Set of directions going from yellow to red
                    yellow_to_red = set(active_phase['yellow'])
                    
                    # Set of directions going from red to green
                    red_to_green = set(next_phase['green'])
                    
                    # Calculate predicted wait time for directions going to red
                    # This is based on the traffic density of the new green directions
                    predicted_wait = self._calculate_wait_time(densities, next_phase['green'])
                    
                    # Change all current yellow to red with predicted wait times
                    for direction in active_phase['yellow']:
                        self.intersections[direction].state = 'red'
                        light_states[direction] = 'red'
                        self.wait_times[direction] = 0  # Reset actual wait time
                        self.predicted_wait_times[direction] = predicted_wait
                        self.initial_wait_times[direction] = predicted_wait
                    
                    # Clear yellow lights
                    active_phase['yellow'] = []
                    
                    # Set new phase's lights to green and reset wait time
                    for direction in next_phase['green']:
                        self.intersections[direction].state = 'green'
                        light_states[direction] = 'green'
                        self.wait_times[direction] = 0
                        self.predicted_wait_times[direction] = 0
                    
                    # Update phase information
                    self.current_phase_index = next_phase_index
                    self.phase_state = "GREEN"
                    self.phase_start_time = current_time
                    
                    # Log the transition
                    self.events.append({
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'type': 'phase_change',
                        'message': f"Changed to {next_phase['name']} phase (GREEN)"
                    })
            
            # Additional safety checks
            # Verify we have active green lights somewhere
            if not any(state == 'green' for state in light_states.values()) and not any(state == 'yellow' for state in light_states.values()):
                logger.warning("No green or yellow lights detected - ensuring green lights are set")
                
                # Turn on green lights for the current phase
                current_phase = self.phases[self.current_phase_index]
                
                # If current phase has no designated green lights, use default
                if not current_phase['green']:
                    if self.current_phase_index == 0:
                        current_phase['green'] = ['north', 'south']
                    else:
                        current_phase['green'] = ['east', 'west']
                
                # Set green lights
                for direction in current_phase['green']:
                    self.intersections[direction].state = 'green'
                    light_states[direction] = 'green'
                    self.wait_times[direction] = 0
                    self.predicted_wait_times[direction] = 0
                
                # Log this correction
                self.events.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'type': 'warning',
                    'message': f"Missing green lights - restored {', '.join(current_phase['green'])} to green"
                })
        
        # Update wait times - they decrease for red lights and reset for green/yellow
        for direction, state in light_states.items():
            if state == 'red':
                # For red lights, decrease the predicted wait time
                if self.predicted_wait_times[direction] > 0:
                    self.predicted_wait_times[direction] -= time_diff
                    # Don't let it go below zero
                    self.predicted_wait_times[direction] = max(0, self.predicted_wait_times[direction])
                
                # Actual elapsed wait time increases (for stats/display)
                self.wait_times[direction] += time_diff
                
                # Cap at 120 seconds for display purposes
                self.wait_times[direction] = min(120, self.wait_times[direction])
            else:
                # Reset wait time for green and yellow lights
                self.wait_times[direction] = 0
                self.predicted_wait_times[direction] = 0
        
        # Handle emergency vehicles
        if random.random() < 0.01:
            emergency_id = f"EMG_{int(time.time())}"
            self.emergency_vehicles.add(emergency_id)
            
            self.events.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'type': 'emergency',
                'message': f"Emergency vehicle {emergency_id} detected"
            })
        
        # Remove emergency vehicles occasionally
        if len(self.emergency_vehicles) > 0 and random.random() < 0.1:
            vehicle_to_remove = random.choice(list(self.emergency_vehicles))
            self.emergency_vehicles.remove(vehicle_to_remove)
            
            self.events.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'type': 'info',
                'message': f"Emergency vehicle {vehicle_to_remove} cleared"
            })
            
        # Keep only last 10 events
        if len(self.events) > 10:
            self.events = self.events[-10:]
            
        self.last_update = current_time
        
        # Count active green signals for stat display
        active_green_count = sum(1 for state in light_states.values() if state == 'green')
        
        return {
            'densities': densities,
            'states': light_states,
            'wait_times': self.predicted_wait_times,  # Return predicted wait times instead of actual elapsed time
            'elapsed_waits': self.wait_times,  # Keep track of actual elapsed wait time separately
            'wait_percentages': {d: 100 * (1 - (self.predicted_wait_times[d] / self.initial_wait_times[d])) if self.initial_wait_times[d] > 0 else 0 
                               for d in self.intersections},  # Percentage of wait time elapsed
            'total_vehicles': self.total_vehicles,
            'emergency_count': len(self.emergency_vehicles),
            'current_phase': self.current_phase_index,
            'phase_info': self.phases[self.current_phase_index],
            'predictions': predictions,
            'events': self.events,
            'active_green': active_green_count,
            'phase_state': self.phase_state
        }
    
    def _get_base_flow_rate(self, hour):
        """Get base vehicle flow rate based on time of day"""
        # Scaling factors for a single intersection with higher minimums
        # Early morning (0-5): Low traffic
        if 0 <= hour < 5:
            return 30  # Increased from 10
        # Morning build-up (5-7): Increasing traffic
        elif 5 <= hour < 7:
            return 50 + (hour - 5) * 25  # Increased from 25+(hour-5)*15
        # Morning rush hour (7-9): Peak traffic
        elif 7 <= hour < 9:
            return 120  # Increased from 60
        # Late morning (9-11): Decreasing traffic
        elif 9 <= hour < 11:
            return 90 - (hour - 9) * 15  # Increased from 50-(hour-9)*10
        # Midday (11-16): Moderate traffic
        elif 11 <= hour < 16:
            return 60  # Increased from 30
        # Evening rush hour (16-19): Peak traffic
        elif 16 <= hour < 19:
            return 130  # Increased from 65
        # Evening wind-down (19-22): Decreasing traffic
        elif 19 <= hour < 22:
            return 80 - (hour - 19) * 15  # Increased from 40-(hour-19)*8
        # Late night (22-24): Low traffic
        else:
            return 35  # Increased from 15

# Initialize traffic management system
traffic_system = TrafficManagementSystem()

# Context processor to add current date to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('config', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('backups', exist_ok=True)
    
    # Start the dashboard update thread
    dashboard_thread = threading.Thread(target=update_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    # Start the scheduler thread
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    
    # Start the web server
    host = '0.0.0.0'
    port = 5000
    print(f"Starting web server on {host}:{port}")
    print(f"Default admin login: admin@example.com / Admin123")
    print("Warning: Running in development mode without SSL. For production, use a proper WSGI server with SSL.")
    
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)