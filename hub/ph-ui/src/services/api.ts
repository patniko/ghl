import axios from 'axios';
import { useAuthStore } from '../stores/authStore';

// Create an axios instance with base URL
const api = axios.create({
  baseURL: 'http://localhost:8000', // Replace with your FastAPI server URL
});

// Authentication services
export const authService = {
  // Login with email and password
  login: async (email: string, password: string) => {
    const response = await api.post('/users/password/login', { email, password });
    return response.data;
  },
  
  // Register a new user
  register: async (userData: { first_name: string, last_name: string, email: string, password: string }) => {
    const response = await api.post('/users/register', userData);
    return response.data;
  },
  
  // Refresh token
  refreshToken: async (refreshToken: string) => {
    const response = await api.post('/users/refresh', { refresh_token: refreshToken });
    return response.data;
  },
  
  // Get current user
  getCurrentUser: async () => {
    const response = await api.get('/users/me');
    return response.data;
  }
};

// Add request interceptor to include auth token
api.interceptors.request.use(
  (config) => {
    const token = useAuthStore.getState().auth.accessToken;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Create a variable to track if a token refresh is in progress
let isRefreshing = false;
let refreshSubscribers = [] as Array<(token: string) => void>;

// Function to add callbacks to the subscriber list
const subscribeTokenRefresh = (callback: (token: string) => void) => {
  refreshSubscribers.push(callback);
};

// Function to notify all subscribers about the new token
const onTokenRefreshed = (token: string) => {
  refreshSubscribers.forEach((callback) => callback(token));
  refreshSubscribers = [];
};

// Function to handle token refresh
const refreshAuthToken = async () => {
  const refreshToken = useAuthStore.getState().auth.refreshToken;
  
  if (!refreshToken) {
    throw new Error('No refresh token available');
  }
  
  try {
    const response = await authService.refreshToken(refreshToken);
    const { access_token, refresh_token, expires_in } = response;
    
    // Update tokens in store with expiration if provided
    useAuthStore.getState().auth.setTokens(access_token, refresh_token, expires_in);
    
    return access_token;
  } catch (error) {
    // If refresh token is invalid, log the user out
    useAuthStore.getState().auth.reset();
    throw error;
  }
};

// Add response interceptor to handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // If the error is 401 (Unauthorized) and we haven't tried to refresh the token yet
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      // If token refresh is already in progress, wait for it to complete
      if (isRefreshing) {
        try {
          // Wait for the token refresh to complete
          const newToken = await new Promise<string>((resolve) => {
            subscribeTokenRefresh((token) => {
              resolve(token);
            });
          });
          
          // Update the Authorization header
          originalRequest.headers.Authorization = `Bearer ${newToken}`;
          
          // Retry the original request
          return api(originalRequest);
        } catch (error) {
          return Promise.reject(error);
        }
      }
      
      // Set refreshing flag
      isRefreshing = true;
      
      try {
        // Refresh the token
        const newToken = await refreshAuthToken();
        
        // Notify all subscribers about the new token
        onTokenRefreshed(newToken);
        
        // Update the Authorization header
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        
        // Reset refreshing flag
        isRefreshing = false;
        
        // Retry the original request
        return api(originalRequest);
      } catch (refreshError) {
        // Reset refreshing flag
        isRefreshing = false;
        
        // Reject all subscribers
        refreshSubscribers = [];
        
        return Promise.reject(refreshError);
      }
    }
    
    return Promise.reject(error);
  }
);

export default api;
