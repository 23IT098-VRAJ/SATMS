// Utility functions
function showAlert(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.querySelector('.container').insertBefore(alertDiv, document.querySelector('.container').firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alertDiv.remove();
    }, 5000);
}

// Form validation
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return true;

    let isValid = true;
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.classList.add('is-invalid');
        } else {
            input.classList.remove('is-invalid');
        }
    });

    return isValid;
}

// API calls
async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
            credentials: 'same-origin'
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(endpoint, options);
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.message || 'An error occurred');
        }

        return result;
    } catch (error) {
        showAlert(error.message, 'danger');
        throw error;
    }
}

// Dashboard charts
function initializeCharts() {
    const chartElements = document.querySelectorAll('[data-chart]');
    
    chartElements.forEach(element => {
        const chartType = element.dataset.chart;
        const chartData = JSON.parse(element.dataset.chartData || '{}');
        
        new Chart(element, {
            type: chartType,
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    });
}

// Real-time updates
function initializeWebSocket() {
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updateDashboard(data);
    };
    
    ws.onclose = function() {
        // Try to reconnect after 5 seconds
        setTimeout(initializeWebSocket, 5000);
    };
}

// Dashboard updates
function updateDashboard(data) {
    // Update statistics
    if (data.stats) {
        Object.entries(data.stats).forEach(([key, value]) => {
            const element = document.getElementById(`stat-${key}`);
            if (element) {
                element.textContent = value;
            }
        });
    }
    
    // Update charts
    if (data.charts) {
        Object.entries(data.charts).forEach(([key, value]) => {
            const chart = Chart.getChart(`chart-${key}`);
            if (chart) {
                chart.data = value;
                chart.update();
            }
        });
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize charts if they exist
    if (document.querySelector('[data-chart]')) {
        initializeCharts();
    }
    
    // Initialize WebSocket if on dashboard
    if (document.querySelector('.dashboard-stats')) {
        initializeWebSocket();
    }
    
    // Add form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!validateForm(form.id)) {
                event.preventDefault();
                showAlert('Please fill in all required fields', 'danger');
            }
        });
    });
}); 