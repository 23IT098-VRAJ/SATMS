import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json
import os
from datetime import datetime
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

class TrafficVisualizer:
    """Visualize traffic simulation and data"""
    
    def __init__(self, config=None, config_path=None, figsize=(15, 10)):
        """Initialize traffic visualizer
        
        Args:
            config: Configuration dictionary with intersection and zones info
            config_path: Path to configuration file
            figsize: Figure size for matplotlib plots
        """
        # Load configuration
        if config is None and config_path is not None:
            self._load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            # Use empty configuration
            self.config = {'zones': {}, 'intersection': {}}
            print("Warning: No configuration provided")
        
        # Extract configuration elements
        self.zones = self.config.get('zones', {})
        self.intersection = self.config.get('intersection', {})
        self.figsize = figsize
        
        # Initialize plot elements
        self.fig = None
        self.axes = None
        self.zone_patches = {}
        self.text_elements = {}
        self.animation = None
        
        # Data storage
        self.traffic_history = {zone_id: [] for zone_id in self.zones}
        self.phase_history = []
        self.speed_history = {zone_id: [] for zone_id in self.zones}
        self.density_history = {zone_id: [] for zone_id in self.zones}
        self.max_history = 100
        
        # Custom color maps for different metrics
        self.setup_colormaps()
        
        # Output directory for saved visualizations
        self.output_dir = "static/visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self.config = {'zones': {}, 'intersection': {}}
    
    def setup_colormaps(self):
        """Setup custom colormaps for different metrics"""
        # Traffic density colormap (green to red)
        self.density_cmap = LinearSegmentedColormap.from_list(
            'density', ['green', 'yellow', 'orange', 'red']
        )
        
        # Speed colormap (red to green)
        self.speed_cmap = LinearSegmentedColormap.from_list(
            'speed', ['red', 'orange', 'yellow', 'green']
        )
        
        # Flow colormap (blue to purple)
        self.flow_cmap = LinearSegmentedColormap.from_list(
            'flow', ['lightblue', 'blue', 'darkblue', 'purple']
        )
    
    def setup_intersection_plot(self):
        """Set up the intersection visualization plot"""
        # Create figure with gridspec for flexible layout
        self.fig = plt.figure(figsize=self.figsize)
        gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
        
        # Main intersection plot
        self.ax_intersection = plt.subplot(gs[0, :2])
        self.ax_intersection.set_title("Intersection Traffic")
        self.ax_intersection.set_xlim(0, 400)
        self.ax_intersection.set_ylim(0, 400)
        self.ax_intersection.set_aspect('equal')
        
        # Traffic history plot
        self.ax_history = plt.subplot(gs[1, :2])
        self.ax_history.set_title("Traffic History")
        self.ax_history.set_xlabel("Time (steps)")
        self.ax_history.set_ylabel("Vehicle Count")
        self.ax_history.grid(True)
        
        # Traffic metrics plot
        self.ax_metrics = plt.subplot(gs[0, 2])
        self.ax_metrics.set_title("Traffic Metrics")
        self.ax_metrics.set_xlim(0, 10)
        self.ax_metrics.set_ylim(0, 10)
        self.ax_metrics.axis('off')
        
        # Prediction plot
        self.ax_prediction = plt.subplot(gs[1, 2])
        self.ax_prediction.set_title("Traffic Prediction")
        self.ax_prediction.set_xlabel("Time (steps)")
        self.ax_prediction.set_ylabel("Value")
        self.ax_prediction.grid(True)
        
        # Draw intersection
        self._draw_intersection(self.ax_intersection)
        
        # Draw zones
        self._draw_zones(self.ax_intersection)
        
        # Create empty lines for each zone in history plot
        self.history_lines = {}
        for zone_id, zone_info in self.zones.items():
            direction = zone_info.get('direction', 'unknown')
            line, = self.ax_history.plot([], [], label=f"{direction.capitalize()} ({zone_id})")
            self.history_lines[zone_id] = line
        
        # Add legend
        self.ax_history.legend(loc='upper left')
        
        # Initialize metrics display
        self._setup_metrics_display()
        
        # Initialize prediction display
        self.prediction_lines = {
            'actual': self.ax_prediction.plot([], [], 'b-', label='Actual')[0],
            'predicted': self.ax_prediction.plot([], [], 'r--', label='Predicted')[0]
        }
        self.ax_prediction.legend(loc='upper left')
        
        plt.tight_layout()
        self.fig.canvas.draw()
        
        return self.fig
    
    def _setup_metrics_display(self):
        """Setup the metrics display area"""
        self.metrics_texts = {}
        
        # Title
        self.metrics_texts['title'] = self.ax_metrics.text(
            5, 9.5, "Traffic Metrics", 
            ha='center', fontsize=12, fontweight='bold'
        )
        
        # Current time
        self.metrics_texts['time'] = self.ax_metrics.text(
            1, 8.5, "Time: 00:00:00", 
            fontsize=10
        )
        
        # Current pattern
        self.metrics_texts['pattern'] = self.ax_metrics.text(
            1, 8, "Pattern: Unknown", 
            fontsize=10
        )
        
        # Phase information
        self.metrics_texts['phase'] = self.ax_metrics.text(
            1, 7.5, "Phase: 0", 
            fontsize=10
        )
        
        # Remaining time
        self.metrics_texts['remaining'] = self.ax_metrics.text(
            1, 7, "Remaining: 0s", 
            fontsize=10
        )
        
        # Average metrics
        y_pos = 6
        for metric in ['Vehicles', 'Density', 'Speed', 'Flow']:
            self.metrics_texts[metric.lower()] = self.ax_metrics.text(
                1, y_pos, f"{metric}: 0", 
                fontsize=10
            )
            y_pos -= 0.5
        
        # Zone-specific metrics
        y_pos = 4
        for zone_id in self.zones:
            self.metrics_texts[f'zone_{zone_id}'] = self.ax_metrics.text(
                1, y_pos, f"Zone {zone_id}: 0 vehicles", 
                fontsize=10
            )
            y_pos -= 0.5
    
    def _draw_intersection(self, ax):
        """Draw the intersection on the given axes"""
        # Draw road outlines
        # Vertical road (north-south)
        ax.add_patch(Rectangle((100, 0), 50, 400, facecolor='gray', edgecolor='black'))
        
        # Horizontal road (east-west)
        ax.add_patch(Rectangle((0, 100), 400, 50, facecolor='gray', edgecolor='black'))
        
        # Draw intersection
        ax.add_patch(Rectangle((100, 100), 50, 50, facecolor='darkgray', edgecolor='black'))
        
        # Add direction labels
        ax.text(125, 380, "North", ha='center')
        ax.text(125, 20, "South", ha='center')
        ax.text(380, 125, "East", ha='center')
        ax.text(20, 125, "West", ha='center')
        
        # Add lane markings
        # North-South lanes
        for x in [112.5, 137.5]:
            ax.plot([x, x], [150, 380], 'w--', linewidth=1)
            ax.plot([x, x], [0, 100], 'w--', linewidth=1)
        
        # East-West lanes
        for y in [112.5, 137.5]:
            ax.plot([0, 100], [y, y], 'w--', linewidth=1)
            ax.plot([150, 400], [y, y], 'w--', linewidth=1)
        
        # Add traffic lights
        self.traffic_lights = {}
        
        # North light
        self.traffic_lights['north'] = Circle((125, 150), 5, facecolor='red', edgecolor='black')
        ax.add_patch(self.traffic_lights['north'])
        
        # South light
        self.traffic_lights['south'] = Circle((125, 100), 5, facecolor='red', edgecolor='black')
        ax.add_patch(self.traffic_lights['south'])
        
        # East light
        self.traffic_lights['east'] = Circle((150, 125), 5, facecolor='red', edgecolor='black')
        ax.add_patch(self.traffic_lights['east'])
        
        # West light
        self.traffic_lights['west'] = Circle((100, 125), 5, facecolor='red', edgecolor='black')
        ax.add_patch(self.traffic_lights['west'])
    
    def _draw_zones(self, ax):
        """Draw traffic zones on the given axes"""
        # Colors for different traffic levels
        colors = {
            'low': 'green',
            'medium': 'yellow',
            'high': 'orange',
            'very_high': 'red'
        }
        
        # Draw each zone
        for zone_id, zone_info in self.zones.items():
            polygon = zone_info.get('polygon', [])
            
            if polygon:
                # Create polygon patch
                patch = Polygon(
                    polygon,
                    facecolor=colors['low'],  # Start with low traffic
                    edgecolor='black',
                    alpha=0.7
                )
                ax.add_patch(patch)
                self.zone_patches[zone_id] = patch
                
                # Calculate center of polygon for text
                center_x = sum(p[0] for p in polygon) / len(polygon)
                center_y = sum(p[1] for p in polygon) / len(polygon)
                
                # Add text for vehicle count
                text = ax.text(
                    center_x, center_y, "0",
                    ha='center', va='center',
                    fontweight='bold',
                    color='white'
                )
                self.text_elements[zone_id] = text
    
    def update_visualization(self, traffic_data, phase_info):
        """Update the visualization with new data
        
        Args:
            traffic_data: Current traffic density data
            phase_info: Current traffic light phase information
            
        Returns:
            Updated figure
        """
        # Update traffic history
        for zone_id, count in traffic_data.get('vehicles', {}).items():
            if zone_id in self.traffic_history:
                self.traffic_history[zone_id].append(count)
                
                # Keep history within limit
                if len(self.traffic_history[zone_id]) > self.max_history:
                    self.traffic_history[zone_id].pop(0)
        
        # Update speed history
        for zone_id, speed in traffic_data.get('traffic_speeds', {}).items():
            if zone_id in self.speed_history:
                self.speed_history[zone_id].append(speed)
                
                # Keep history within limit
                if len(self.speed_history[zone_id]) > self.max_history:
                    self.speed_history[zone_id].pop(0)
        
        # Update density history
        for zone_id, density in traffic_data.get('traffic_density', {}).items():
            if zone_id in self.density_history:
                self.density_history[zone_id].append(density)
                
                # Keep history within limit
                if len(self.density_history[zone_id]) > self.max_history:
                    self.density_history[zone_id].pop(0)
        
        # Update phase history
        current_phase = phase_info.get('id', 0)
        self.phase_history.append(current_phase)
        if len(self.phase_history) > self.max_history:
            self.phase_history.pop(0)
        
        # Update zone colors and text
        for zone_id, patch in self.zone_patches.items():
            vehicle_count = traffic_data.get('vehicles', {}).get(zone_id, 0)
            density = traffic_data.get('traffic_density', {}).get(zone_id, 0)
            
            # Update color based on traffic density
            if density < 0.25:
                color = 'green'  # low
            elif density < 0.5:
                color = 'yellow'  # medium
            elif density < 0.75:
                color = 'orange'  # high
            else:
                color = 'red'  # very high
            
            patch.set_facecolor(color)
            
            # Update text
            if zone_id in self.text_elements:
                self.text_elements[zone_id].set_text(f"{vehicle_count}")
                
                # Make text more visible on different backgrounds
                self.text_elements[zone_id].set_color('black' if color in ['yellow', 'green'] else 'white')
        
        # Update traffic history plot
        x = np.arange(self.max_history)
        for zone_id, line in self.history_lines.items():
            data = self.traffic_history[zone_id]
            if data:
                # Pad data if needed
                padded_data = data + [np.nan] * (self.max_history - len(data))
                line.set_data(x[:len(data)], data)
        
        # Update axes limits for history plot
        self.ax_history.relim()
        self.ax_history.autoscale_view()
        
        # Highlight current phase
        directions = phase_info.get('directions', [])
        
        # Reset all traffic lights
        for direction, light in self.traffic_lights.items():
            light.set_facecolor('red')
        
        # Highlight zones with green light
        for zone_id, patch in self.zone_patches.items():
            patch.set_linewidth(1)
            
            zone_direction = self.zones[zone_id].get('direction', 'unknown')
            if zone_direction in directions:
                # This zone has green light
                patch.set_linewidth(3)
                patch.set_edgecolor('lime')
                
                # Update traffic light
                if zone_direction in self.traffic_lights:
                    self.traffic_lights[zone_direction].set_facecolor('green')
            else:
                # Red light
                patch.set_edgecolor('black')
        
        # Update metrics display
        self._update_metrics_display(traffic_data, phase_info)
        
        # Add phase info to plot
        remaining_time = phase_info.get('remaining_time', 0)
        phase_text = f"Phase: {current_phase}, Time: {remaining_time:.1f}s"
        self.ax_intersection.set_title(f"Intersection Traffic\n{phase_text}")
        
        # Draw canvas
        self.fig.canvas.draw_idle()
        
        return self.fig
    
    def _update_metrics_display(self, traffic_data, phase_info):
        """Update the metrics display with current data"""
        # Update time
        if 'timestamp' in traffic_data:
            try:
                dt = datetime.fromisoformat(traffic_data['timestamp'].replace('Z', '+00:00'))
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = f"{traffic_data.get('hour', 0):02d}:{traffic_data.get('minute', 0):02d}:00"
        else:
            time_str = f"{traffic_data.get('hour', 0):02d}:{traffic_data.get('minute', 0):02d}:00"
        
        self.metrics_texts['time'].set_text(f"Time: {time_str}")
        
        # Update pattern
        pattern = traffic_data.get('pattern', 'Unknown')
        self.metrics_texts['pattern'].set_text(f"Pattern: {pattern}")
        
        # Update phase information
        current_phase = phase_info.get('id', 0)
        self.metrics_texts['phase'].set_text(f"Phase: {current_phase}")
        
        # Update remaining time
        remaining_time = phase_info.get('remaining_time', 0)
        self.metrics_texts['remaining'].set_text(f"Remaining: {remaining_time:.1f}s")
        
        # Update average metrics
        vehicles = traffic_data.get('vehicles', {})
        densities = traffic_data.get('traffic_density', {})
        speeds = traffic_data.get('traffic_speeds', {})
        
        avg_vehicles = sum(vehicles.values()) / max(1, len(vehicles))
        avg_density = sum(densities.values()) / max(1, len(densities))
        avg_speed = sum(speeds.values()) / max(1, len(speeds))
        avg_flow = avg_density * avg_speed * 0.1  # Simple flow calculation
        
        self.metrics_texts['vehicles'].set_text(f"Vehicles: {avg_vehicles:.1f}")
        self.metrics_texts['density'].set_text(f"Density: {avg_density:.2f}")
        self.metrics_texts['speed'].set_text(f"Speed: {avg_speed:.1f} mph")
        self.metrics_texts['flow'].set_text(f"Flow: {avg_flow:.1f} veh/min")
        
        # Update zone-specific metrics
        for zone_id in self.zones:
            vehicle_count = vehicles.get(zone_id, 0)
            density = densities.get(zone_id, 0)
            speed = speeds.get(zone_id, 0)
            
            self.metrics_texts[f'zone_{zone_id}'].set_text(
                f"Zone {zone_id}: {vehicle_count} veh, {speed:.1f} mph"
            )
    
    def update_prediction_plot(self, actual_data, predicted_data, zone_id=None):
        """Update the prediction plot with actual and predicted data
        
        Args:
            actual_data: List of actual traffic values
            predicted_data: List of predicted traffic values
            zone_id: Optional zone ID for title
        """
        # Clear previous data
        self.prediction_lines['actual'].set_data([], [])
        self.prediction_lines['predicted'].set_data([], [])
        
        # Set new data
        if actual_data and len(actual_data) > 0:
            x_actual = np.arange(len(actual_data))
            self.prediction_lines['actual'].set_data(x_actual, actual_data)
        
        if predicted_data and len(predicted_data) > 0:
            x_pred = np.arange(len(predicted_data))
            self.prediction_lines['predicted'].set_data(x_pred, predicted_data)
        
        # Update title if zone_id provided
        if zone_id:
            self.ax_prediction.set_title(f"Traffic Prediction - Zone {zone_id}")
        
        # Update axes limits
        self.ax_prediction.relim()
        self.ax_prediction.autoscale_view()
    
    def start_animation(self, update_func, interval=100):
        """Start real-time animation
        
        Args:
            update_func: Function that returns (traffic_data, phase_info)
            interval: Update interval in milliseconds
            
        Returns:
            Animation object
        """
        if self.fig is None:
            self.setup_intersection_plot()
        
        def animate(i):
            traffic_data, phase_info = update_func()
            self.update_visualization(traffic_data, phase_info)
            return list(self.zone_patches.values()) + list(self.text_elements.values())
        
        self.animation = FuncAnimation(
            self.fig, animate, interval=interval, blit=False
        )
        
        return self.animation
    
    def save_snapshot(self, filename=None):
        """Save current visualization as image
        
        Args:
            filename: Output filename (if None, auto-generate based on timestamp)
        """
        if self.fig:
            if filename is None:
                # Generate filename based on timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.output_dir, f"traffic_snapshot_{timestamp}.png")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Snapshot saved to {filename}")
            return filename
        else:
            print("No figure to save")
            return None
    
    def create_heatmap(self, data, title="Traffic Density Heatmap", save_path=None):
        """Create a heatmap visualization of traffic density
        
        Args:
            data: 2D array of traffic density values
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(data, cmap='hot', interpolation='nearest')
        ax.set_title(title)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Vehicle Count')
        
        # Add grid
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_predictions(self, actual, predicted, zone_id=None, metric="traffic", save_path=None):
        """Visualize traffic predictions against actual values
        
        Args:
            actual: Array of actual traffic values
            predicted: Array of predicted traffic values
            zone_id: Zone ID for the title
            metric: Metric type (traffic, speed, density)
            save_path: Optional path to save the figure
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual values
        ax.plot(actual, 'b-', label='Actual')
        
        # Plot predicted values
        ax.plot(predicted, 'r--', label='Predicted')
        
        # Set title and labels
        zone_str = f"Zone {zone_id}" if zone_id else "All Zones"
        metric_str = metric.capitalize()
        ax.set_title(f"{metric_str} Prediction for {zone_str}")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel(metric_str)
        
        # Add legend and grid
        ax.legend()
        ax.grid(True)
        
        # Calculate and display metrics
        if len(actual) > 0 and len(predicted) > 0:
            # Calculate error metrics
            mse = np.mean((np.array(actual) - np.array(predicted))**2)
            mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
            
            # Add text with metrics
            metrics_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}"
            ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_traffic_dashboard(self, traffic_data, save_path=None):
        """Create a comprehensive traffic dashboard
        
        Args:
            traffic_data: Dictionary with traffic metrics
            save_path: Optional path to save the figure
            
        Returns:
            Figure object
        """
        # Create figure with gridspec
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 3)
        
        # Extract data
        vehicles = traffic_data.get('vehicles', {})
        densities = traffic_data.get('traffic_density', {})
        speeds = traffic_data.get('traffic_speeds', {})
        
        # Calculate flow rates
        flow_rates = {}
        for zone_id in vehicles.keys():
            density = densities.get(zone_id, 0)
            speed = speeds.get(zone_id, 0)
            flow_rates[zone_id] = density * speed * 0.1
        
        # 1. Traffic volume by zone
        ax1 = fig.add_subplot(gs[0, 0])
        self._create_bar_chart(ax1, vehicles, "Traffic Volume by Zone", "Vehicles")
        
        # 2. Traffic density by zone
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_bar_chart(ax2, densities, "Traffic Density by Zone", "Density")
        
        # 3. Traffic speed by zone
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_bar_chart(ax3, speeds, "Traffic Speed by Zone", "Speed (mph)")
        
        # 4. Flow rate by zone
        ax4 = fig.add_subplot(gs[1, 0])
        self._create_bar_chart(ax4, flow_rates, "Traffic Flow by Zone", "Flow Rate")
        
        # 5. Intersection diagram
        ax5 = fig.add_subplot(gs[1, 1:])
        self._create_intersection_diagram(ax5, traffic_data)
        
        # 6. Time series history
        ax6 = fig.add_subplot(gs[2, :])
        self._create_time_series(ax6, traffic_data)
        
        # Add overall title
        timestamp = traffic_data.get('timestamp', datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = timestamp
        
        fig.suptitle(f"Traffic Dashboard - {time_str}", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def _create_bar_chart(self, ax, data, title, ylabel):
        """Create a bar chart on the given axes"""
        zone_ids = list(data.keys())
        values = [data[zone_id] for zone_id in zone_ids]
        
        bars = ax.bar(zone_ids, values)
        
        # Color bars based on values
        for i, bar in enumerate(bars):
            value = values[i]
            max_value = max(values) if values else 1
            normalized = value / max_value
            
            if ylabel.lower() == "speed (mph)":
                # For speed, higher is better (green)
                color = self.speed_cmap(normalized)
            elif "flow" in ylabel.lower():
                # For flow, use flow colormap
                color = self.flow_cmap(normalized)
            else:
                # For density and volume, lower is better (green)
                color = self.density_cmap(normalized)
            
            bar.set_color(color)
        
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Zone")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    def _create_intersection_diagram(self, ax, traffic_data):
        """Create a simplified intersection diagram on the given axes"""
        # Draw intersection
        ax.add_patch(Rectangle((4, 4), 2, 2, facecolor='darkgray', edgecolor='black'))
        
        # Draw roads
        ax.add_patch(Rectangle((4, 0), 2, 4, facecolor='gray', edgecolor='black'))  # South
        ax.add_patch(Rectangle((4, 6), 2, 4, facecolor='gray', edgecolor='black'))  # North
        ax.add_patch(Rectangle((0, 4), 4, 2, facecolor='gray', edgecolor='black'))  # West
        ax.add_patch(Rectangle((6, 4), 4, 2, facecolor='gray', edgecolor='black'))  # East
        
        # Draw zones with traffic
        zone_positions = {
            'zone_1': {'position': (5, 8), 'direction': 'north'},
            'zone_2': {'position': (5, 2), 'direction': 'south'},
            'zone_3': {'position': (8, 5), 'direction': 'east'},
            'zone_4': {'position': (2, 5), 'direction': 'west'}
        }
        
        # Draw traffic lights
        current_phase = traffic_data.get('current_phase', 0)
        phase_info = traffic_data.get('phase_info', {'directions': []})
        green_directions = phase_info.get('directions', [])
        
        for zone_id, zone_info in zone_positions.items():
            pos = zone_info['position']
            direction = zone_info['direction']
            
            # Get traffic data
            vehicle_count = traffic_data.get('vehicles', {}).get(zone_id, 0)
            density = traffic_data.get('traffic_density', {}).get(zone_id, 0)
            
            # Determine color based on density
            if density < 0.25:
                color = 'green'
            elif density < 0.5:
                color = 'yellow'
            elif density < 0.75:
                color = 'orange'
            else:
                color = 'red'
            
            # Draw zone
            if direction == 'north':
                rect = Rectangle((pos[0]-1, pos[1]-1), 2, 2, facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                ax.text(pos[0], pos[1], str(vehicle_count), ha='center', va='center', fontweight='bold')
            elif direction == 'south':
                rect = Rectangle((pos[0]-1, pos[1]-1), 2, 2, facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                ax.text(pos[0], pos[1], str(vehicle_count), ha='center', va='center', fontweight='bold')
            elif direction == 'east':
                rect = Rectangle((pos[0]-1, pos[1]-1), 2, 2, facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                ax.text(pos[0], pos[1], str(vehicle_count), ha='center', va='center', fontweight='bold')
            elif direction == 'west':
                rect = Rectangle((pos[0]-1, pos[1]-1), 2, 2, facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                ax.text(pos[0], pos[1], str(vehicle_count), ha='center', va='center', fontweight='bold')
            
            # Draw traffic light
            light_positions = {
                'north': (5, 6),
                'south': (5, 4),
                'east': (6, 5),
                'west': (4, 5)
            }
            
            light_pos = light_positions[direction]
            light_color = 'green' if direction in green_directions else 'red'
            light = Circle(light_pos, 0.3, facecolor=light_color, edgecolor='black')
            ax.add_patch(light)
        
        # Set limits and title
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title("Current Intersection State")
        ax.set_aspect('equal')
        ax.axis('off')
    
    def _create_time_series(self, ax, traffic_data):
        """Create a time series plot of recent traffic history"""
        # Get history data
        histories = []
        labels = []
        
        # Only plot if we have history
        if hasattr(self, 'traffic_history') and self.traffic_history:
            for zone_id, history in self.traffic_history.items():
                if history:
                    histories.append(history)
                    zone_direction = self.zones.get(zone_id, {}).get('direction', 'unknown')
                    labels.append(f"{zone_direction.capitalize()} ({zone_id})")
        
        # Plot each zone's history
        for i, (history, label) in enumerate(zip(histories, labels)):
            ax.plot(history, label=label)
        
        # Add phase changes as vertical lines
        if hasattr(self, 'phase_history') and self.phase_history:
            phase_changes = [i for i in range(1, len(self.phase_history)) 
                            if self.phase_history[i] != self.phase_history[i-1]]
            
            for change in phase_changes:
                ax.axvline(x=change, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title("Recent Traffic History")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Vehicle Count")
        ax.legend(loc='upper left')
        ax.grid(True)