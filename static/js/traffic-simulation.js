// Traffic simulation functions
document.addEventListener('DOMContentLoaded', function() {
    console.log('Traffic simulation script loaded');
    
    // Make sure elements exist before trying to use them
    if (!document.getElementById('remaining-time') || 
        !document.getElementById('current-phase')) {
        console.error('Required dashboard elements not found');
        return;
    }
    
    console.log('Found remaining-time element:', !!document.getElementById('remaining-time'));
    console.log('Found current-phase element:', !!document.getElementById('current-phase'));
    console.log('Found traffic lights:', !!document.querySelector('.traffic-light'));
    
    // Simulate traffic light changes
    let countdown = 15;
    let currentState = 'NS_RED'; // Start with North-South Red (East-West Green)
    
    // Update countdown immediately
    document.getElementById('remaining-time').textContent = `${countdown}s`;
    
    const timerInterval = setInterval(() => {
        countdown--;
        
        // Update the timer display
        document.getElementById('remaining-time').textContent = `${countdown}s`;
        
        if (countdown <= 0) {
            // Change lights
            document.querySelectorAll('.light.active').forEach(light => {
                light.classList.remove('active');
            });
            
            // Toggle between states
            if (currentState === 'NS_RED') {
                // Change to North-South Green, East-West Red
                document.querySelector('.north .green').classList.add('active');
                document.querySelector('.south .green').classList.add('active');
                document.querySelector('.east .red').classList.add('active');
                document.querySelector('.west .red').classList.add('active');
                document.getElementById('current-phase').textContent = 'North-South Green, East-West Red';
                currentState = 'NS_GREEN';
            } else {
                // Change to North-South Red, East-West Green
                document.querySelector('.north .red').classList.add('active');
                document.querySelector('.south .red').classList.add('active');
                document.querySelector('.east .green').classList.add('active');
                document.querySelector('.west .green').classList.add('active');
                document.getElementById('current-phase').textContent = 'North-South Red, East-West Green';
                currentState = 'NS_RED';
            }
            
            // Reset countdown
            countdown = 15;
        }
    }, 1000);
    
    // Clean up interval when page changes
    window.addEventListener('beforeunload', function() {
        clearInterval(timerInterval);
    });
});