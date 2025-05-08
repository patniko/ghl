import { create } from 'zustand'
import { authService } from '../services/api'
import { organizationService } from '../services/organizationService'

const ACCESS_TOKEN = 'accessToken'
const REFRESH_TOKEN = 'refreshToken'
const TOKEN_EXPIRY = 'tokenExpiry' // Store token expiration time

interface Organization {
  id: number
  name: string
  slug: string
  description?: string
  created_at: string
  updated_at: string
  role?: string
}

interface AuthUser {
  id: number
  first_name: string
  last_name: string
  email: string
  email_verified: boolean
  picture?: string
  organizations: Organization[]
}

interface AuthState {
  auth: {
    user: AuthUser | null
    setUser: (user: AuthUser | null) => void
    accessToken: string
    refreshToken: string
    tokenExpiry: number | null
    isInitializing: boolean
    isInitialized: boolean
    setTokens: (accessToken: string, refreshToken: string, expiresIn?: number) => void
    resetTokens: () => void
    reset: () => void
    login: (email: string, password: string) => Promise<AuthUser>
    register: (userData: { first_name: string, last_name: string, email: string, password: string }) => Promise<unknown>
    logout: () => void
    createOrganization: (name: string, slug: string, description?: string) => Promise<Organization>
    ensurePersonalOrg: () => Promise<Organization>
    initAuth: () => Promise<void>
    refreshTokenIfNeeded: () => Promise<boolean>
    isTokenExpired: () => boolean
    isTokenExpiringSoon: (minutesThreshold?: number) => boolean
  }
}

export const useAuthStore = create<AuthState>()((set, get) => {
  const storedAccessToken = localStorage.getItem(ACCESS_TOKEN)
  const storedRefreshToken = localStorage.getItem(REFRESH_TOKEN)
  const storedTokenExpiry = localStorage.getItem(TOKEN_EXPIRY)
  
  return {
    auth: {
      user: null,
      setUser: (user) =>
        set((state) => ({ ...state, auth: { ...state.auth, user } })),
      accessToken: storedAccessToken ? JSON.parse(storedAccessToken) : '',
      refreshToken: storedRefreshToken ? JSON.parse(storedRefreshToken) : '',
      tokenExpiry: storedTokenExpiry ? JSON.parse(storedTokenExpiry) : null,
      isInitializing: false,
      isInitialized: false,
      setTokens: (accessToken, refreshToken, expiresIn = 86400) => // Default to 24 hours
        set((state) => {
          const expiryTime = Date.now() + expiresIn * 1000
          localStorage.setItem(ACCESS_TOKEN, JSON.stringify(accessToken))
          localStorage.setItem(REFRESH_TOKEN, JSON.stringify(refreshToken))
          localStorage.setItem(TOKEN_EXPIRY, JSON.stringify(expiryTime))
          return { 
            ...state, 
            auth: { 
              ...state.auth, 
              accessToken, 
              refreshToken,
              tokenExpiry: expiryTime
            } 
          }
        }),
      resetTokens: () =>
        set((state) => {
          localStorage.removeItem(ACCESS_TOKEN)
          localStorage.removeItem(REFRESH_TOKEN)
          localStorage.removeItem(TOKEN_EXPIRY)
          return { 
            ...state, 
            auth: { 
              ...state.auth, 
              accessToken: '', 
              refreshToken: '',
              tokenExpiry: null
            } 
          }
        }),
      reset: () =>
        set((state) => {
          localStorage.removeItem(ACCESS_TOKEN)
          localStorage.removeItem(REFRESH_TOKEN)
          localStorage.removeItem(TOKEN_EXPIRY)
          return {
            ...state,
            auth: { 
              ...state.auth, 
              user: null, 
              accessToken: '', 
              refreshToken: '',
              tokenExpiry: null
            },
          }
        }),
      login: async (email, password) => {
        const response = await authService.login(email, password)
        const { access_token, refresh_token, expires_in } = response
        
        // Use the expires_in from the server if available, otherwise use default (24 hours)
        get().auth.setTokens(access_token, refresh_token, expires_in)
        
        // Fetch user details
        const userResponse = await authService.getCurrentUser()
        get().auth.setUser(userResponse)
        
        // Ensure the user has a personal organization
        console.log('[DEBUG] Ensuring personal organization exists after login')
        try {
          await get().auth.ensurePersonalOrg()
          console.log('[DEBUG] Personal organization ensured after login')
        } catch (orgError) {
          console.error('[DEBUG] Error ensuring personal organization after login:', orgError)
          // Continue with login even if organization creation fails
        }
        
        return userResponse
      },
      register: async (userData) => {
        const response = await authService.register(userData)
        return response
      },
      logout: () => {
        get().auth.reset()
      },
      createOrganization: async (name, slug, description) => {
        try {
          console.log(`[DEBUG] Creating organization in authStore: ${name}, ${slug}`);
          
          // Call the API to create the organization
          const newOrg = await organizationService.createOrganization({
            name,
            slug,
            description
          });
          
          // Update the user's organizations
          const user = get().auth.user;
          if (user) {
            // Add role property to match the expected Organization interface in authStore
            const orgWithRole = {
              ...newOrg,
              role: 'admin'
            };
            
            const updatedUser = {
              ...user,
              organizations: [...user.organizations, orgWithRole]
            };
            get().auth.setUser(updatedUser);
          }
          
          return {
            ...newOrg,
            role: 'admin'
          };
        } catch (error) {
          console.error('[DEBUG] Error creating organization:', error);
          throw error;
        }
      },
      ensurePersonalOrg: async () => {
        try {
          console.log('[DEBUG] Ensuring personal organization exists');
          const user = get().auth.user;
          
          // If user has organizations, return the first one
          if (user && user.organizations && user.organizations.length > 0) {
            console.log('[DEBUG] User already has organizations:', user.organizations);
            return user.organizations[0];
          }
          
          console.log('[DEBUG] No organizations found, creating personal organization');
          
          // Create a personal organization
          const personalOrg = await get().auth.createOrganization(
            'Personal',
            'personal',
            'Your personal organization'
          );
          
          // Set as active organization in localStorage
          localStorage.setItem('org_slug', personalOrg.slug);
          
          console.log('[DEBUG] Personal organization created:', personalOrg);
          return personalOrg;
        } catch (error) {
          console.error('[DEBUG] Error ensuring personal organization:', error);
          throw error;
        }
      },
      
      isTokenExpired: () => {
        const { tokenExpiry } = get().auth
        if (!tokenExpiry) return true
        return Date.now() > tokenExpiry
      },
      
      isTokenExpiringSoon: (minutesThreshold = 30) => {
        const { tokenExpiry } = get().auth
        if (!tokenExpiry) return true
        // Check if token will expire within the next X minutes
        return Date.now() > tokenExpiry - (minutesThreshold * 60 * 1000)
      },
      
      refreshTokenIfNeeded: async () => {
        try {
          const { refreshToken, isTokenExpiringSoon } = get().auth
          
          // If token is expiring soon and we have a refresh token, refresh it
          if (isTokenExpiringSoon() && refreshToken) {
            console.log('Token expiring soon, refreshing...')
            const response = await authService.refreshToken(refreshToken)
            const { access_token, refresh_token } = response
            
            // Update tokens in store (with default 24 hour expiry if not provided)
            get().auth.setTokens(access_token, refresh_token)
            return true
          }
          return false
        } catch (error) {
          console.error('Error refreshing token:', error)
          // If there's an error refreshing the token, reset the auth state
          get().auth.reset()
          return false
        }
      },
      
      initAuth: async () => {
        try {
          // Set initializing state
          set((state) => ({ 
            ...state, 
            auth: { 
              ...state.auth, 
              isInitializing: true 
            } 
          }))
          
          const { accessToken, refreshToken, isTokenExpired, refreshTokenIfNeeded } = get().auth
          
          // If token is expired but we have a refresh token, try to refresh it
          if (isTokenExpired() && refreshToken) {
            console.log('Token expired, attempting to refresh')
            const refreshed = await refreshTokenIfNeeded()
            if (!refreshed) {
              console.log('Failed to refresh token')
              // Set initialized state even if refresh failed
              set((state) => ({ 
                ...state, 
                auth: { 
                  ...state.auth, 
                  isInitializing: false,
                  isInitialized: true 
                } 
              }))
              return
            }
          }
          
          // If we have tokens, fetch the user (even if we already have a user object, refresh it)
          if (accessToken && refreshToken) {
            console.log('Initializing auth: Found tokens, fetching user')
            try {
              // Fetch user details
              const userResponse = await authService.getCurrentUser()
              get().auth.setUser(userResponse)
              
              // Ensure the user has a personal organization
              console.log('Ensuring personal organization exists during init')
              try {
                await get().auth.ensurePersonalOrg()
                console.log('Personal organization ensured during init')
              } catch (orgError) {
                console.error('Error ensuring personal organization during init:', orgError)
                // Continue with initialization even if organization creation fails
              }
              
              // Set up periodic token refresh (every 5 minutes)
              const refreshInterval = setInterval(() => {
                get().auth.refreshTokenIfNeeded()
                  .then(refreshed => {
                    if (refreshed) {
                      console.log('Token refreshed successfully')
                    }
                  })
                  .catch(error => console.error('Error in periodic token refresh:', error))
              }, 5 * 60 * 1000) // 5 minutes
              
              // Clean up interval on window unload
              window.addEventListener('beforeunload', () => {
                clearInterval(refreshInterval)
              })
            } catch (error) {
              console.error('Error fetching user during auth initialization:', error)
              // If we can't fetch the user, reset auth state
              get().auth.reset()
            }
          } else {
            console.log('Initializing auth: No tokens found')
          }
          
          // Set initialized state
          set((state) => ({ 
            ...state, 
            auth: { 
              ...state.auth, 
              isInitializing: false,
              isInitialized: true 
            } 
          }))
        } catch (error) {
          console.error('Error initializing auth:', error)
          // If there's an error (like expired token), reset the auth state
          get().auth.reset()
          
          // Set initialized state even if there was an error
          set((state) => ({ 
            ...state, 
            auth: { 
              ...state.auth, 
              isInitializing: false,
              isInitialized: true 
            } 
          }))
        }
      }
    },
  }
})

export const useAuth = () => useAuthStore((state) => state.auth)
