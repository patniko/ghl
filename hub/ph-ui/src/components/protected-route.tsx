import { ReactNode, useEffect, useState } from 'react'
import { useNavigate, useLocation } from '@tanstack/react-router'
import { useAuth } from '@/stores/authStore'

interface ProtectedRouteProps {
  children: ReactNode
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const { accessToken, user, isInitializing, isInitialized, initAuth } = useAuth()
  const [isLoading, setIsLoading] = useState(true)
  const navigate = useNavigate()
  const location = useLocation()

  // Initialize auth if not already initialized
  useEffect(() => {
    const initialize = async () => {
      if (!isInitialized && !isInitializing) {
        console.log('Protected route: initializing auth')
        await initAuth()
      }
      setIsLoading(false)
    }
    
    initialize()
  }, [isInitialized, isInitializing, initAuth])

  // Handle redirection after auth is initialized
  useEffect(() => {
    // Only check authentication after initialization is complete
    if (!isLoading && isInitialized) {
      // If there's no access token or user, redirect to login
      if (!accessToken || !user) {
        // Check if we're already on the sign-in page to prevent infinite redirect loops
        if (!location.pathname.includes('/sign-in')) {
          // Save the current location to redirect back after login
          // Use a string path instead of an object
          const currentPath = typeof location.pathname === 'string' 
            ? location.pathname 
            : '/'
          const currentSearch = typeof location.search === 'string'
            ? location.search
            : ''
          const redirect = encodeURIComponent(currentPath + currentSearch)
          console.log('Setting redirect to:', redirect)
          navigate({ to: '/sign-in', search: { redirect } })
        } else {
          console.log('Already on sign-in page, not setting redirect')
        }
      }
    }
  }, [accessToken, user, navigate, location, isLoading, isInitialized])

  // Show loading state while initializing
  if (isLoading || isInitializing) {
    // You could return a loading spinner here if desired
    return null
  }

  // If authenticated, render the children
  if (accessToken && user) {
    return <>{children}</>
  }

  // Return null while redirecting
  return null
}
