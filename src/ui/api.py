from flask import Flask, jsonify, request
import threading
import json
import os

class TrafficAPI:
    """REST API for the traffic management system"""
    
    def __init__(self, port=5000):
        """Initialize the API server
        
        Args:
            port: Port number for the API server
        """
        self.app = Flask(__name__)
        self.port = port
        self.traffic_data = {}
        self.phase_info = {}
        self.thread = None
        self.running = False
        
        # Define API routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes"""
        
        @self.app.route('/api/traffic', methods=['GET'])
        def get_traffic():
            """Get current traffic data"""
            return jsonify(self.traffic_data)
        
        @self.app.route('/api/phase', methods=['GET'])
        def get_phase():
            """Get current traffic light phase"""
            return jsonify(self.phase_info)
        
        @self.app.route('/api/phase', methods=['POST'])
        def set_phase():
            """Manually set traffic light phase"""
            data = request.json
            
            # This would trigger a callback to the controller
            # For now, just store the requested phase
            requested_phase = data.get('phase_id')
            
            return jsonify({'status': 'success', 'message': f'Requested phase change to {requested_phase}'})
    
    def start(self):
        """Start the API server in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_server)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the API server"""
        self.running = False
        # Shutdown would be handled properly in a production environment
    
    def _run_server(self):
        """Run the Flask server"""
        self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
    
    def update_data(self, traffic_data, phase_info):
        """Update API data
        
        Args:
            traffic_data: Current traffic density data
            phase_info: Current traffic light phase information
        """
        self.traffic_data = traffic_data
        self.phase_info = phase_info