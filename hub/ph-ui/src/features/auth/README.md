# Authentication Implementation

This directory contains the authentication implementation for the application. It uses the FastAPI backend for authentication and provides sign-in and sign-up functionality.

## Features

- User registration with email and password
- User login with email and password
- Protected routes that require authentication
- Automatic token refresh
- Logout functionality

## Components

- `sign-in/components/user-auth-form.tsx`: Form for user login
- `sign-up/components/sign-up-form.tsx`: Form for user registration
- `components/protected-route.tsx`: Component to protect routes that require authentication

## Services

- `services/api.ts`: API service for making requests to the backend
  - Includes authentication endpoints
  - Handles token refresh
  - Adds authorization headers to requests

## State Management

- `stores/authStore.ts`: Zustand store for managing authentication state
  - Stores user information
  - Stores access and refresh tokens
  - Provides login, register, and logout functions

## Usage

### Login

```tsx
import { useAuth } from '@/stores/authStore'

function LoginComponent() {
  const { login } = useAuth()
  
  const handleLogin = async (email, password) => {
    try {
      await login(email, password)
      // Redirect or show success message
    } catch (error) {
      // Handle error
    }
  }
  
  // ...
}
```

### Registration

```tsx
import { useAuth } from '@/stores/authStore'

function RegisterComponent() {
  const { register } = useAuth()
  
  const handleRegister = async (userData) => {
    try {
      await register(userData)
      // Redirect or show success message
    } catch (error) {
      // Handle error
    }
  }
  
  // ...
}
```

### Protected Routes

```tsx
import { ProtectedRoute } from '@/components/protected-route'

function PrivateComponent() {
  return (
    <ProtectedRoute>
      {/* Your protected content here */}
    </ProtectedRoute>
  )
}
```

### Logout

```tsx
import { useAuth } from '@/stores/authStore'

function LogoutButton() {
  const { logout } = useAuth()
  
  const handleLogout = () => {
    logout()
    // Redirect or show success message
  }
  
  return <button onClick={handleLogout}>Logout</button>
}
```

## Backend API Endpoints

The authentication implementation uses the following FastAPI endpoints:

- `POST /register`: Register a new user
- `POST /password/login`: Login with email and password
- `POST /refresh`: Refresh access token
- `GET /me`: Get current user information
