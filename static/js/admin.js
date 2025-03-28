document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('create-backup-btn').addEventListener('click', function() {
        this.disabled = true;
        this.textContent = 'Creating backup...';
        
        fetch('/api/backup/create', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            const message = document.getElementById('message');
            if (data.error) {
                message.textContent = data.error;
                message.className = 'message error';
            } else {
                message.textContent = data.message;
                message.className = 'message success';
                setTimeout(function() {
                    window.location.reload();
                }, 2000);
            }
            message.style.display = 'block';
            this.disabled = false;
            this.textContent = 'Create New Backup';
        })
        .catch(error => {
            console.error('Error:', error);
            this.disabled = false;
            this.textContent = 'Create New Backup';
        });
    });
    
    // Setup restore buttons
    document.querySelectorAll('.restore-btn').forEach(button => {
        button.addEventListener('click', function() {
            const file = this.getAttribute('data-file');
            if (confirm('Are you sure you want to restore from this backup? Current data will be replaced.')) {
                this.disabled = true;
                this.textContent = 'Restoring...';
                
                fetch('/api/backup/restore', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ file })
                })
                .then(response => response.json())
                .then(data => {
                    const message = document.getElementById('message');
                    if (data.error) {
                        message.textContent = data.error;
                        message.className = 'message error';
                    } else {
                        message.textContent = data.message;
                        message.className = 'message success';
                    }
                    message.style.display = 'block';
                    this.disabled = false;
                    this.textContent = 'Restore';
                })
                .catch(error => {
                    console.error('Error:', error);
                    this.disabled = false;
                    this.textContent = 'Restore';
                });
            }
        });
    });
    
    // Setup delete buttons
    document.querySelectorAll('.delete-btn').forEach(button => {
        button.addEventListener('click', function() {
            const file = this.getAttribute('data-file');
            if (confirm('Are you sure you want to delete this backup?')) {
                this.disabled = true;
                this.textContent = 'Deleting...';
                
                fetch('/api/backup/delete', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ file })
                })
                .then(response => response.json())
                .then(data => {
                    const message = document.getElementById('message');
                    if (data.error) {
                        message.textContent = data.error;
                        message.className = 'message error';
                    } else {
                        message.textContent = data.message;
                        message.className = 'message success';
                        // Remove the row from the table
                        this.closest('tr').remove();
                    }
                    message.style.display = 'block';
                })
                .catch(error => {
                    console.error('Error:', error);
                    this.disabled = false;
                    this.textContent = 'Delete';
                });
            }
        });
    });
});
