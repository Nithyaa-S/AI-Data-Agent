// API Configuration
export const API_CONFIG = {
  // Use the Render backend URL in production
  // Fall back to localhost for development
  BASE_URL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  
  // API endpoints
  ENDPOINTS: {
    ASK: "/api/ask",
    UPLOAD: "/api/upload",
    HEALTH: "/health"
  },
  
  // Timeout for API requests in milliseconds
  TIMEOUT: 30000,
  
  // Default headers
  HEADERS: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  }
};

// Helper function to get full API URL
export const getApiUrl = (endpoint) => {
  // Remove leading slash if present
  const path = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
  return `${API_CONFIG.BASE_URL}/${path}`;
};
