// Main shared frontend script
document.addEventListener('DOMContentLoaded', () => {
  // Highlight active navbar link
  const path = window.location.pathname;
  const navMap = {
    '/': 'nav-home',
    '/recommend': 'nav-recommend',
    '/predict': 'nav-predict',
    '/trends': 'nav-trends',
    '/performance': 'nav-performance'
  };
  const activeId = navMap[path] || 'nav-home';
  const activeLink = document.getElementById(activeId);
  if (activeLink) {
    activeLink.classList.add('active');
  }
});

// Debounce helper
function debounce(func, wait = 300) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

