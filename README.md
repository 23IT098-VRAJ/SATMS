# Smart Traffic Management System (SATMS)

A comprehensive AI-powered traffic management solution that utilizes computer vision, machine learning, and real-time data analysis to optimize traffic flow and reduce congestion in urban environments.

![SATMS Dashboard](static/images/dashboard_preview.png)

## 🚦 Key Features

- **Real-time Traffic Monitoring**: Continuous monitoring and analysis of traffic conditions
- **AI-driven Traffic Light Control**: Intelligent timing adjustments based on traffic density
- **Predictive Traffic Analysis**: Machine learning models to forecast traffic patterns
- **Emergency Vehicle Priority**: Automatic priority routing for emergency vehicles
- **Interactive Dashboard**: Web-based control center for monitoring and management
- **RESTful API**: Easy integration with external systems and applications
- **Data Backup & Recovery**: Robust system for backup and restoration of configuration and data
- **User Management**: Secure multi-user system with role-based access control

## 📋 Requirements

- Python 3.8 or higher
- Flask and web dependencies
- Data processing libraries
- Machine learning framework
- System requirements:
  - 2GB RAM minimum (4GB recommended)
  - 1GB free disk space
  - Modern web browser

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-organization/smart-traffic-management-system.git
   cd smart-traffic-management-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the system**
   ```bash
   python main.py --init
   ```

## 🚀 Quick Start

1. **Start the SATMS server**
   ```bash
   python main.py
   ```

2. **Access the dashboard**
   - Open your web browser and navigate to `http://localhost:5000`
   - Login with default admin credentials:
     - Email: `admin@example.com`
     - Password: `Admin123`
   - **Important**: Change the default password immediately after first login

3. **Configure the system**
   - Navigate to the Settings panel
   - Adjust traffic parameters
   - Configure notification settings
   - Set up monitoring points

## 🏗️ Project Structure

```
SATMS/
├── src/                  # Source code
│   ├── utils/            # Utility functions
│   ├── ui/               # User interface components
│   ├── simulation/       # Traffic simulation modules
│   ├── processing/       # Data processing modules
│   ├── prediction/       # Traffic prediction algorithms
│   ├── monitoring/       # System monitoring components
│   └── control/          # Traffic control logic
├── config/               # Configuration files
├── data/                 # Data storage
├── templates/            # HTML templates for web interface
├── static/               # Static web assets (CSS, JS, images)
├── logs/                 # System logs
├── models/               # Saved ML models
├── tests/                # Test suite
├── backups/              # Backup storage
├── main.py               # Main application entry point
└── requirements.txt      # Python dependencies
```

## 🔍 Configuration

The system can be configured through multiple methods:

1. **Web Interface**: Most settings accessible through the admin dashboard
2. **Configuration Files**: Edit JSON files in the `config/` directory:
   - `system.json`: Core system parameters
   - `traffic.json`: Traffic pattern configurations
   - `security.json`: Security and authentication settings
3. **Environment Variables**: Override settings with environment variables prefixed with `SATMS_`

## 🛠️ Development Guide

### Setup Development Environment

1. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests**
   ```bash
   pytest
   ```

3. **Code style and formatting**
   ```bash
   black .
   flake8
   ```

### Adding New Features

1. Create a feature branch (`git checkout -b feature/your-feature`)
2. Implement your changes
3. Write tests in the `tests/` directory
4. Update documentation as needed
5. Submit a pull request

## 📊 API Documentation

The SATMS provides a RESTful API for integration with other systems:

- **Authentication**: `/api/login` (POST)
- **Traffic Data**: `/api/traffic/current` (GET)
- **Predictions**: `/api/traffic/predict` (GET, POST)
- **System Control**: `/api/system/control` (POST)
- **Configuration**: `/api/config` (GET, PUT)

For detailed API documentation, visit `/api/docs` in the running application.

## 🔒 Security

- **Authentication**: Email and password-based with session management
- **Authorization**: Role-based access control (Admin, Operator, Viewer)
- **Data Protection**: Encrypted storage for sensitive information
- **API Security**: Token-based authentication for API access

## 🔄 Backup and Recovery

The system includes built-in functionality for backup and recovery:

1. **Manual Backup**: `/admin/backup` in the web interface
2. **Scheduled Backups**: Configurable intervals for automatic backups
3. **Restore**: System can be restored from any previous backup point

## 📝 Logging and Monitoring

- Comprehensive logging system with configurable verbosity
- Performance monitoring for system resources
- Traffic event logging for historical analysis
- Error tracking and notification system

## 🤝 Contributing

We welcome contributions to the Smart Traffic Management System:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the test suite to ensure everything works
5. Submit a pull request

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Support and Community

- **Issue Tracking**: Submit bugs and feature requests via the GitHub issue tracker
- **Discussion Forum**: Join our community discussions at [forum.satms.org](https://forum.satms.org)
- **Documentation**: Extended documentation available at [docs.satms.org](https://docs.satms.org)

## 🙏 Acknowledgments

- This project builds on research from Smart City initiatives
- Special thanks to all contributors and supporters
- Uses technology from the following open-source projects:
  - Flask web framework
  - TensorFlow and NumPy for data analysis
  - Matplotlib and Plotly for visualization
- Inspired by smart city initiatives
- Built with modern AI and computer vision technologies
