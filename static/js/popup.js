// Budgeteer Notification System
window.BudgeteerNotifications = {
    showSuccess: function(message) {
        this._showToast('success', '✅ ' + message);
    },
    
    showWarning: function(message) {
        this._showToast('warning', '⚠️ ' + message);
    },
    
    showDanger: function(message) {
        this._showToast('danger', '❌ ' + message);
    },
    
    showInfo: function(message) {
        this._showToast('info', 'ℹ️ ' + message);
    },
    
    _showToast: function(type, message) {
        // Create toast container if it doesn't exist
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        `;
        
        // Add to container and show
        container.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove from DOM after hiding
        toast.addEventListener('hidden.bs.toast', function() {
            toast.remove();
        });
    }
};

// Fallback for auto-converting flash messages
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert.d-none');
    alerts.forEach(function(alert) {
        const category = alert.classList.contains('alert-success') ? 'success' :
                        alert.classList.contains('alert-warning') ? 'warning' :
                        alert.classList.contains('alert-danger') ? 'danger' : 'info';
        const message = alert.textContent.trim();
        
        if (window.BudgeteerNotifications && window.BudgeteerNotifications['show' + category.charAt(0).toUpperCase() + category.slice(1)]) {
            window.BudgeteerNotifications['show' + category.charAt(0).toUpperCase() + category.slice(1)](message);
        }
        alert.remove();
    });
});