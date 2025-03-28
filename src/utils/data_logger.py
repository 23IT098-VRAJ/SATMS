import os
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time

class TrafficDataLogger:
    """Logger for recording traffic patterns and system performance"""
    
    def __init__(self, log_dir="logs"):
        """Initialize the data logger
        
        Args:
            log_dir: Directory to store logs
        """
        self.log_dir = log_dir
        self.traffic_log_file = None
        self.performance_log_file = None
        self.event_log_file = None
        self.start_time = time.time()
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "traffic"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "performance"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "events"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "reports"), exist_ok=True)
        
        # Initialize log files with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.traffic_log_file = os.path.join(log_dir, "traffic", f"traffic_log_{timestamp}.csv")
        self.performance_log_file = os.path.join(log_dir, "performance", f"performance_log_{timestamp}.csv")
        self.event_log_file = os.path.join(log_dir, "events", f"event_log_{timestamp}.csv")
        
        # Initialize CSV headers
        self._initialize_log_files()
        
    def _initialize_log_files(self):
        """Initialize log files with headers"""
        # Traffic log headers
        with open(self.traffic_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'simulation_time', 'hour', 'day_of_week', 
                'zone_id', 'direction', 'vehicle_count', 'traffic_pattern'
            ])
            
        # Performance log headers
        with open(self.performance_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'simulation_time', 'phase_id', 'phase_duration',
                'avg_wait_time', 'max_wait_time', 'throughput', 'queue_length'
            ])
            
        # Event log headers
        with open(self.event_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'simulation_time', 'event_type', 'description', 'details'
            ])
    
    def log_traffic_state(self, traffic_state, timestamp=None):
        """Log current traffic state
        
        Args:
            traffic_state: Dictionary with current traffic state
            timestamp: Optional timestamp (defaults to current time)
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        simulation_time = traffic_state.get('time', 0)
        hour = traffic_state.get('hour', 0)
        day_of_week = traffic_state.get('day_of_week', 0)
        pattern = traffic_state.get('pattern', 'unknown')
        
        with open(self.traffic_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Log each zone's traffic data
            vehicles = traffic_state.get('vehicles', {})
            for zone_id, count in vehicles.items():
                # Get direction for this zone (if available)
                direction = "unknown"  # Default
                # This would require access to zone config - placeholder for now
                
                writer.writerow([
                    timestamp, simulation_time, hour, day_of_week,
                    zone_id, direction, count, pattern
                ])
    
    def log_performance(self, phase_info, traffic_metrics, timestamp=None):
        """Log system performance metrics
        
        Args:
            phase_info: Current traffic light phase information
            traffic_metrics: Dictionary with performance metrics
            timestamp: Optional timestamp (defaults to current time)
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        simulation_time = time.time() - self.start_time
        phase_id = phase_info.get('id', 0)
        phase_duration = phase_info.get('total_time', 0)
        
        avg_wait_time = traffic_metrics.get('avg_wait_time', 0)
        max_wait_time = traffic_metrics.get('max_wait_time', 0)
        throughput = traffic_metrics.get('throughput', 0)
        queue_length = traffic_metrics.get('queue_length', 0)
        
        with open(self.performance_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, simulation_time, phase_id, phase_duration,
                avg_wait_time, max_wait_time, throughput, queue_length
            ])
    
    def log_event(self, event_type, description, details=None, timestamp=None):
        """Log significant events
        
        Args:
            event_type: Type of event (e.g., 'phase_change', 'congestion', 'error')
            description: Brief description of the event
            details: Additional details (dictionary)
            timestamp: Optional timestamp (defaults to current time)
        """
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
        simulation_time = time.time() - self.start_time
        
        # Convert details dictionary to JSON string if provided
        details_str = json.dumps(details) if details else ""
        
        with open(self.event_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, simulation_time, event_type, description, details_str
            ])
    
    def generate_traffic_report(self, report_type="daily", output_format="html"):
        """Generate a report from logged data
        
        Args:
            report_type: Type of report ('daily', 'hourly', 'summary')
            output_format: Output format ('html', 'pdf', 'csv')
            
        Returns:
            Path to the generated report file
        """
        # Load traffic data
        try:
            traffic_data = pd.read_csv(self.traffic_log_file)
        except Exception as e:
            print(f"Error loading traffic data: {e}")
            return None
            
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.log_dir, "reports", f"{report_type}_report_{timestamp}.{output_format}")
        
        if report_type == "daily":
            self._generate_daily_report(traffic_data, report_file, output_format)
        elif report_type == "hourly":
            self._generate_hourly_report(traffic_data, report_file, output_format)
        elif report_type == "summary":
            self._generate_summary_report(traffic_data, report_file, output_format)
        else:
            print(f"Unknown report type: {report_type}")
            return None
            
        return report_file
    
    def _generate_daily_report(self, data, report_file, output_format):
        """Generate daily traffic report"""
        # Example implementation - would be customized based on your needs
        if output_format == "html":
            # Group by day and calculate statistics
            if 'timestamp' in data.columns:
                data['date'] = pd.to_datetime(data['timestamp']).dt.date
                daily_stats = data.groupby(['date', 'zone_id']).agg({
                    'vehicle_count': ['mean', 'max', 'sum']
                }).reset_index()
                
                # Generate HTML report
                html_content = f"""
                <html>
                <head>
                    <title>Daily Traffic Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #2c3e50; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                        th {{ background-color: #3498db; color: white; }}
                        tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>Daily Traffic Report</h1>
                    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <h2>Daily Traffic Statistics</h2>
                    <table>
                        <tr>
                            <th>Date</th>
                            <th>Zone ID</th>
                            <th>Average Vehicles</th>
                            <th>Maximum Vehicles</th>
                            <th>Total Vehicles</th>
                        </tr>
                """
                
                for _, row in daily_stats.iterrows():
                    html_content += f"""
                        <tr>
                            <td>{row['date']}</td>
                            <td>{row['zone_id']}</td>
                            <td>{row[('vehicle_count', 'mean')]:.2f}</td>
                            <td>{row[('vehicle_count', 'max')]:.0f}</td>
                            <td>{row[('vehicle_count', 'sum')]:.0f}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </body>
                </html>
                """
                
                with open(report_file, 'w') as f:
                    f.write(html_content)
            
        elif output_format == "csv":
            # Just save the filtered/aggregated data
            data.to_csv(report_file, index=False)
    
    def _generate_hourly_report(self, data, report_file, output_format):
        """Generate hourly traffic report"""
        # Implementation similar to daily report but grouped by hour
        pass
    
    def _generate_summary_report(self, data, report_file, output_format):
        """Generate summary traffic report"""
        # Implementation for summary statistics
        pass
    
    def plot_traffic_patterns(self, zone_ids=None, start_time=None, end_time=None, 
                             save_path=None, show_plot=True):
        """Plot traffic patterns from logged data
        
        Args:
            zone_ids: List of zone IDs to include (None for all)
            start_time: Start time for filtering data
            end_time: End time for filtering data
            save_path: Path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Load traffic data
            data = pd.read_csv(self.traffic_log_file)
            
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Filter by time range if specified
            if start_time:
                data = data[data['timestamp'] >= pd.to_datetime(start_time)]
            if end_time:
                data = data[data['timestamp'] <= pd.to_datetime(end_time)]
                
            # Filter by zone IDs if specified
            if zone_ids:
                data = data[data['zone_id'].isin(zone_ids)]
                
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Group by timestamp and zone_id, then plot
            for zone_id, group in data.groupby('zone_id'):
                ax.plot(group['timestamp'], group['vehicle_count'], 
                       label=f'Zone {zone_id}', marker='o', markersize=3)
                
            ax.set_xlabel('Time')
            ax.set_ylabel('Vehicle Count')
            ax.set_title('Traffic Patterns Over Time')
            ax.legend()
            ax.grid(True)
            
            # Format x-axis for better readability
            fig.autofmt_xdate()
            
            # Save plot if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
            # Show plot if requested
            if show_plot:
                plt.show()
                
            return fig
            
        except Exception as e:
            print(f"Error plotting traffic patterns: {e}")
            return None