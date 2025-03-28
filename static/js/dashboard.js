// Create traffic chart
document.addEventListener('DOMContentLoaded', function() {
    // Sample data for chart
    const trafficData = {
        labels: Array.from({length: 20}, (_, i) => `${i*5}m ago`).reverse(),
        datasets: [
            {
                label: 'North',
                data: Array.from({length: 20}, () => Math.floor(Math.random() * 20) + 5),
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
            },
            {
                label: 'South',
                data: Array.from({length: 20}, () => Math.floor(Math.random() * 20) + 5),
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
            },
            {
                label: 'East',
                data: Array.from({length: 20}, () => Math.floor(Math.random() * 20) + 5),
                borderColor: 'rgba(255, 206, 86, 1)',
                backgroundColor: 'rgba(255, 206, 86, 0.2)',
            },
            {
                label: 'West',
                data: Array.from({length: 20}, () => Math.floor(Math.random() * 20) + 5),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
            }
        ]
    };
    
    const ctx = document.getElementById('traffic-chart').getContext('2d');
    const trafficChart = new Chart(ctx, {
        type: 'line',
        data: trafficData,
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Vehicle Count'
                    }
                }
            }
        }
    });
    
    // Apply strategy button
    document.getElementById('apply-strategy').addEventListener('click', function() {
        const strategy = document.getElementById('strategy-select').value;
        alert(`Strategy changed to: ${strategy}`);
        // In a real implementation, this would call an API endpoint
    });
    
    // Simulate traffic metrics updates
    setInterval(() => {
        document.getElementById('avg-wait-time').textContent = `${(Math.random() * 20 + 5).toFixed(1)}s`;
        document.getElementById('max-wait-time').textContent = `${(Math.random() * 30 + 15).toFixed(1)}s`;
        document.getElementById('throughput').textContent = `${Math.floor(Math.random() * 20 + 30)} veh/min`;
        document.getElementById('queue-length').textContent = `${Math.floor(Math.random() * 20 + 5)} vehicles`;
        
        // Update chart data
        trafficChart.data.datasets.forEach(dataset => {
            dataset.data.shift();
            dataset.data.push(Math.floor(Math.random() * 20) + 5);
        });
        trafficChart.update();
    }, 5000);
});