import threading
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    import cv2
    from PIL import Image, ImageTk
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available, camera view will be disabled")

class Dashboard:
    """Enhanced dashboard UI for the traffic management system"""
    
    def __init__(self, config):
        """Initialize the dashboard
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.running = False
        self.thread = None
        self.root = None
        
        # Data storage
        self.traffic_data = {}
        self.phase_info = {}
        self.current_frame = None
        self.performance_metrics = {}
        
        # History tracking
        self.max_history = 100
        self.traffic_history = {zone_id: [] for zone_id in config['zones']}
        self.phase_history = []
        self.wait_time_history = []
        self.throughput_history = []
        
        # UI elements
        self.traffic_labels = {}
        self.phase_label = None
        self.countdown_label = None
        self.camera_label = None
        self.strategy_var = None
        self.strategy_combobox = None
        self.traffic_canvas = None
        self.wait_time_canvas = None
        self.visualization_tabs = None
        
        # Visualization settings
        self.visualization_mode = "traffic"  # traffic, wait_time, heatmap
        self.update_interval = 100  # ms
        
        # Controller reference (will be set externally)
        self.controller = None
    
    def set_controller(self, controller):
        """Set reference to the traffic light controller
        
        Args:
            controller: TrafficLightController instance
        """
        self.controller = controller
    
    def start(self):
        """Start the dashboard in a separate thread"""
        self.running = True
        self.thread = threading.Thread(target=self._run_dashboard)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the dashboard"""
        self.running = False
        if self.root:
            self.root.quit()
    
    def update(self, traffic_data, phase_info, frame=None, performance_metrics=None):
        """Update dashboard with new data
        
        Args:
            traffic_data: Current traffic density data
            phase_info: Current traffic light phase information
            frame: Optional camera frame for display
            performance_metrics: Optional performance metrics
        """
        self.traffic_data = traffic_data
        self.phase_info = phase_info
        self.current_frame = frame
        
        if performance_metrics:
            self.performance_metrics = performance_metrics
        
        # Update history
        for zone_id, count in traffic_data.items():
            if zone_id in self.traffic_history:
                self.traffic_history[zone_id].append(count)
                # Keep history within limit
                if len(self.traffic_history[zone_id]) > self.max_history:
                    self.traffic_history[zone_id].pop(0)
        
        if phase_info:
            self.phase_history.append(phase_info.get('id', 0))
            if len(self.phase_history) > self.max_history:
                self.phase_history.pop(0)
        
        # Update performance history
        if performance_metrics:
            wait_time = performance_metrics.get('avg_wait_time', 0)
            throughput = performance_metrics.get('throughput', 0)
            
            self.wait_time_history.append(wait_time)
            if len(self.wait_time_history) > self.max_history:
                self.wait_time_history.pop(0)
                
            self.throughput_history.append(throughput)
            if len(self.throughput_history) > self.max_history:
                self.throughput_history.pop(0)
    
    def _run_dashboard(self):
        """Run the dashboard UI"""
        self.root = tk.Tk()
        self.root.title("Smart Traffic Management System")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        
        # Create UI layout
        self._create_ui()
        
        # Start update loop
        self._update_ui()
        
        # Run Tkinter event loop
        self.root.mainloop()
    
    def _create_ui(self):
        """Create the dashboard UI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top panel - controls and status
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Strategy selection
        ttk.Label(top_panel, text="Optimization Strategy:").pack(side=tk.LEFT, padx=(0, 5))
        self.strategy_var = tk.StringVar(value="proportional")
        self.strategy_combobox = ttk.Combobox(
            top_panel, 
            textvariable=self.strategy_var,
            values=["fixed", "proportional", "webster", "adaptive"],
            state="readonly",
            width=15
        )
        self.strategy_combobox.pack(side=tk.LEFT, padx=(0, 20))
        self.strategy_combobox.bind("<<ComboboxSelected>>", self._on_strategy_change)
        
        # Phase info
        phase_frame = ttk.LabelFrame(top_panel, text="Traffic Light Phase", padding=5)
        phase_frame.pack(side=tk.LEFT, padx=10)
        
        self.phase_label = ttk.Label(phase_frame, text="Current Phase: None")
        self.phase_label.pack(side=tk.LEFT, padx=5)
        
        self.countdown_label = ttk.Label(phase_frame, text="Time Remaining: 0s")
        self.countdown_label.pack(side=tk.LEFT, padx=5)
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(top_panel, text="Performance Metrics", padding=5)
        metrics_frame.pack(side=tk.RIGHT)
        
        self.wait_time_label = ttk.Label(metrics_frame, text="Avg Wait Time: 0.0s")
        self.wait_time_label.pack(side=tk.LEFT, padx=5)
        
        self.throughput_label = ttk.Label(metrics_frame, text="Throughput: 0 veh/min")
        self.throughput_label.pack(side=tk.LEFT, padx=5)
        
        self.queue_label = ttk.Label(metrics_frame, text="Queue Length: 0")
        self.queue_label.pack(side=tk.LEFT, padx=5)
        
        # Create notebook for different visualizations
        self.visualization_tabs = ttk.Notebook(main_frame)
        self.visualization_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Traffic visualization tab
        traffic_tab = ttk.Frame(self.visualization_tabs)
        self.visualization_tabs.add(traffic_tab, text="Traffic Visualization")
        
        # Split traffic tab into two panels
        traffic_left_panel = ttk.Frame(traffic_tab)
        traffic_left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        traffic_right_panel = ttk.Frame(traffic_tab)
        traffic_right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Traffic density visualization (left panel)
        density_frame = ttk.LabelFrame(traffic_left_panel, text="Traffic Density", padding=10)
        density_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create zone labels
        for zone_id, zone_info in self.config['zones'].items():
            direction = zone_info.get('direction', 'unknown')
            frame = ttk.Frame(density_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label_text = f"{direction.capitalize()} ({zone_id}):"
            ttk.Label(frame, text=label_text, width=15).pack(side=tk.LEFT)
            
            # Progress bar for visual representation
            progress = ttk.Progressbar(frame, length=200, mode='determinate', maximum=30)
            progress.pack(side=tk.LEFT, padx=5)
            
            # Value label
            value_label = ttk.Label(frame, text="0 vehicles")
            value_label.pack(side=tk.LEFT, padx=5)
            
            self.traffic_labels[zone_id] = {
                'progress': progress,
                'label': value_label
            }
        
        # Traffic history plot (right panel)
        fig_traffic = plt.Figure(figsize=(6, 4), dpi=100)
        self.traffic_ax = fig_traffic.add_subplot(111)
        self.traffic_ax.set_title('Traffic Density History')
        self.traffic_ax.set_xlabel('Time Steps')
        self.traffic_ax.set_ylabel('Vehicle Count')
        self.traffic_ax.grid(True)
        
        self.traffic_canvas = FigureCanvasTkAgg(fig_traffic, traffic_right_panel)
        self.traffic_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Performance visualization tab
        performance_tab = ttk.Frame(self.visualization_tabs)
        self.visualization_tabs.add(performance_tab, text="Performance Metrics")
        
        # Split performance tab into two panels
        perf_left_panel = ttk.Frame(performance_tab)
        perf_left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        perf_right_panel = ttk.Frame(performance_tab)
        perf_right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Wait time history plot (left panel)
        fig_wait = plt.Figure(figsize=(6, 4), dpi=100)
        self.wait_ax = fig_wait.add_subplot(111)
        self.wait_ax.set_title('Average Wait Time History')
        self.wait_ax.set_xlabel('Time Steps')
        self.wait_ax.set_ylabel('Wait Time (s)')
        self.wait_ax.grid(True)
        
        self.wait_canvas = FigureCanvasTkAgg(fig_wait, perf_left_panel)
        self.wait_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Throughput history plot (right panel)
        fig_throughput = plt.Figure(figsize=(6, 4), dpi=100)
        self.throughput_ax = fig_throughput.add_subplot(111)
        self.throughput_ax.set_title('Throughput History')
        self.throughput_ax.set_xlabel('Time Steps')
        self.throughput_ax.set_ylabel('Vehicles per Minute')
        self.throughput_ax.grid(True)
        
        self.throughput_canvas = FigureCanvasTkAgg(fig_throughput, perf_right_panel)
        self.throughput_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Intersection visualization tab
        intersection_tab = ttk.Frame(self.visualization_tabs)
        self.visualization_tabs.add(intersection_tab, text="Intersection View")
        
        # Create intersection visualization
        fig_intersection = plt.Figure(figsize=(8, 6), dpi=100)
        self.intersection_ax = fig_intersection.add_subplot(111)
        self.intersection_ax.set_title('Intersection Status')
        self.intersection_ax.set_xlim(0, 400)
        self.intersection_ax.set_ylim(0, 400)
        self.intersection_ax.set_aspect('equal')
        
        self.intersection_canvas = FigureCanvasTkAgg(fig_intersection, intersection_tab)
        self.intersection_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Draw intersection
        self._draw_intersection()
        
        # Status bar
        status_bar = ttk.Label(
            self.root, 
            text="Smart Traffic Management System - Simulation Mode", 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _on_strategy_change(self, event):
        """Handle strategy selection change"""
        selected_strategy = self.strategy_var.get()
        if self.controller:
            if self.controller.set_optimization_strategy(selected_strategy):
                print(f"Changed optimization strategy to: {selected_strategy}")
            else:
                print(f"Failed to change strategy to: {selected_strategy}")
    
    def _draw_intersection(self):
        """Draw the intersection layout"""
        # Clear previous drawings
        self.intersection_ax.clear()
        
        # Draw road outlines
        # Vertical road (north-south)
        self.intersection_ax.add_patch(
            plt.Rectangle((100, 0), 50, 400, facecolor='gray', edgecolor='black')
        )
        
        # Horizontal road (east-west)
        self.intersection_ax.add_patch(
            plt.Rectangle((0, 100), 400, 50, facecolor='gray', edgecolor='black')
        )
        
        # Draw intersection
        self.intersection_ax.add_patch(
            plt.Rectangle((100, 100), 50, 50, facecolor='darkgray', edgecolor='black')
        )
        
        # Add direction labels
        self.intersection_ax.text(125, 380, "North", ha='center')
        self.intersection_ax.text(125, 20, "South", ha='center')
        self.intersection_ax.text(380, 125, "East", ha='center')
        self.intersection_ax.text(20, 125, "West", ha='center')
        
        # Draw zones
        zone_colors = {
            'zone_1': 'lightgreen',  # North
            'zone_2': 'lightblue',   # South
            'zone_3': 'lightyellow', # East
            'zone_4': 'lightpink'    # West
        }
        
        for zone_id, zone_info in self.config['zones'].items():
            if 'polygon' in zone_info:
                polygon = zone_info['polygon']
                color = zone_colors.get(zone_id, 'lightgray')
                self.intersection_ax.add_patch(
                    plt.Polygon(polygon, facecolor=color, alpha=0.7, edgecolor='black')
                )
        
        self.intersection_ax.set_title('Intersection Status')
        self.intersection_canvas.draw()
    
    def _update_intersection_visualization(self):
        """Update the intersection visualization based on current state"""
        # Clear previous drawings
        self.intersection_ax.clear()
        
        # Redraw base intersection
        self._draw_intersection()
        
        # Update traffic density in zones
        for zone_id, count in self.traffic_data.items():
            # Find zone polygon
            zone_info = self.config['zones'].get(zone_id)
            if zone_info and 'polygon' in zone_info:
                polygon = zone_info['polygon']
                
                # Calculate center for text
                center_x = sum(p[0] for p in polygon) / len(polygon)
                center_y = sum(p[1] for p in polygon) / len(polygon)
                
                # Add text with vehicle count
                self.intersection_ax.text(
                    center_x, center_y, 
                    str(int(count)), 
                    ha='center', va='center', 
                    fontweight='bold', fontsize=12
                )
        
        # Highlight active phase
        if self.phase_info:
            phase_id = self.phase_info.get('id')
            directions = self.phase_info.get('directions', [])
            
            # Highlight directions with green light
            for direction in directions:
                if direction == 'north':
                    self.intersection_ax.add_patch(
                        plt.Rectangle((110, 150), 30, 250, facecolor='green', alpha=0.3)
                    )
                elif direction == 'south':
                    self.intersection_ax.add_patch(
                        plt.Rectangle((110, 0), 30, 100, facecolor='green', alpha=0.3)
                    )
                elif direction == 'east':
                    self.intersection_ax.add_patch(
                        plt.Rectangle((150, 110), 250, 30, facecolor='green', alpha=0.3)
                    )
                elif direction == 'west':
                    self.intersection_ax.add_patch(
                        plt.Rectangle((0, 110), 100, 30, facecolor='green', alpha=0.3)
                    )
            
            # Add phase info to title
            remaining_time = self.phase_info.get('remaining_time', 0)
            self.intersection_ax.set_title(f'Intersection Status - Phase {phase_id} ({remaining_time:.1f}s remaining)')
        
        # Update canvas
        self.intersection_canvas.draw()
    
    def _update_ui(self):
        """Update UI elements with current data"""
        if not self.running:
            return
            
        # Update traffic density displays
        for zone_id, components in self.traffic_labels.items():
            count = self.traffic_data.get(zone_id, 0)
            
            # Update progress bar
            components['progress']['value'] = min(30, count)  # Cap at maximum
            
            # Update label
            components['label'].config(text=f"{count:.1f} vehicles")
        
        # Update phase information
        if self.phase_info:
            phase_id = self.phase_info.get('id', 'None')
            directions = ', '.join(self.phase_info.get('directions', []))
            remaining_time = self.phase_info.get('remaining_time', 0)
            
            self.phase_label.config(text=f"Phase: {phase_id} ({directions})")
            self.countdown_label.config(text=f"Remaining: {remaining_time:.1f}s")
        
        # Update performance metrics
        if self.performance_metrics:
            wait_time = self.performance_metrics.get('avg_wait_time', 0)
            throughput = self.performance_metrics.get('throughput', 0)
            queue_length = self.performance_metrics.get('queue_length', 0)
            
            self.wait_time_label.config(text=f"Avg Wait: {wait_time:.1f}s")
            self.throughput_label.config(text=f"Throughput: {throughput} veh/min")
            self.queue_label.config(text=f"Queue: {queue_length} vehicles")
        
        # Update traffic history plot
        self.traffic_ax.clear()
        
        # Plot each zone's history
        for zone_id, history in self.traffic_history.items():
            if history:
                zone_info = self.config['zones'].get(zone_id, {})
                direction = zone_info.get('direction', 'unknown')
                label = f"{direction.capitalize()} ({zone_id})"
                self.traffic_ax.plot(history, label=label)
        
        self.traffic_ax.set_title('Traffic Density History')
        self.traffic_ax.set_xlabel('Time Steps')
        self.traffic_ax.set_ylabel('Vehicle Count')
        self.traffic_ax.grid(True)
        self.traffic_ax.legend(loc='upper right')
        
        self.traffic_canvas.draw()
        
        # Update wait time history plot
        self.wait_ax.clear()
        if self.wait_time_history:
            self.wait_ax.plot(self.wait_time_history, 'r-', label='Avg Wait Time')
            self.wait_ax.set_title('Wait Time History')
            self.wait_ax.set_xlabel('Time Steps')
            self.wait_ax.set_ylabel('Wait Time (s)')
            self.wait_ax.grid(True)
            self.wait_ax.legend()
            
        self.wait_canvas.draw()
        
        # Update throughput history plot
        self.throughput_ax.clear()
        if self.throughput_history:
            self.throughput_ax.plot(self.throughput_history, 'g-', label='Throughput')
            self.throughput_ax.set_title('Throughput History')
            self.throughput_ax.set_xlabel('Time Steps')
            self.throughput_ax.set_ylabel('Vehicles per Minute')
            self.throughput_ax.grid(True)
            self.throughput_ax.legend()
            
        self.throughput_canvas.draw()
        
        # Update intersection visualization
        self._update_intersection_visualization()
        
        # Schedule next update
        if self.root:
            self.root.after(self.update_interval, self._update_ui)