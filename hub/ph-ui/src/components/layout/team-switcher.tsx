import * as React from 'react'
import { useCallback } from 'react'
import { ChevronsUpDown, Plus, Building } from 'lucide-react'
import { useNavigate } from '@tanstack/react-router'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuShortcut,
} from '@/components/ui/dropdown-menu'
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from '@/components/ui/sidebar'
import { useAuth } from '@/stores/authStore'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/hooks/use-toast'

interface Organization {
  id: number
  name: string
  slug: string
  description?: string
}

export function TeamSwitcher() {
  const { isMobile } = useSidebar()
  const auth = useAuth()
  const navigate = useNavigate()
  const { toast } = useToast()
  
  // State for organizations
  const [organizations, setOrganizations] = React.useState<Organization[]>([])
  const [activeOrg, setActiveOrg] = React.useState<Organization | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  
  // State for create organization dialog
  const [isCreateOrgOpen, setIsCreateOrgOpen] = React.useState(false)
  const [newOrgName, setNewOrgName] = React.useState('')
  const [newOrgSlug, setNewOrgSlug] = React.useState('')
  const [isCreatingOrg, setIsCreatingOrg] = React.useState(false)

  // Create a personal organization for the user
  const createPersonalOrg = useCallback(async () => {
    try {
      const personalOrg = await auth.ensurePersonalOrg()
      
      setOrganizations([personalOrg])
      setActiveOrg(personalOrg)
      localStorage.setItem('org_slug', personalOrg.slug)
    } catch (error) {
      console.error('Error creating personal organization:', error)
      toast({
        title: 'Error',
        description: 'Failed to create personal organization',
        variant: 'destructive',
      })
    }
  }, [auth, toast])

  // Fetch organizations on component mount
  React.useEffect(() => {
    const fetchOrganizations = async () => {
      setIsLoading(true)
      try {
        // In a real app, you would fetch organizations from the API
        // For now, we'll use the user's organizations from the auth store
        if (auth.user && auth.user.organizations) {
          setOrganizations(auth.user.organizations)
          
          // Set active organization from localStorage or use the first one
          const storedOrgSlug = localStorage.getItem('org_slug')
          const activeOrg = storedOrgSlug 
            ? auth.user.organizations.find(org => org.slug === storedOrgSlug)
            : auth.user.organizations[0]
          
          if (activeOrg) {
            setActiveOrg(activeOrg)
            localStorage.setItem('org_slug', activeOrg.slug)
          } else if (auth.user.organizations.length > 0) {
            setActiveOrg(auth.user.organizations[0])
            localStorage.setItem('org_slug', auth.user.organizations[0].slug)
          } else {
            // If no organizations, create a personal one
            createPersonalOrg()
          }
        } else {
          // If no user or no organizations, create a personal one
          createPersonalOrg()
        }
      } catch (error) {
        console.error('Error fetching organizations:', error)
        toast({
          title: 'Error',
          description: 'Failed to fetch organizations',
          variant: 'destructive',
        })
      } finally {
        setIsLoading(false)
      }
    }

    fetchOrganizations()
  }, [auth.user, toast, createPersonalOrg])


  // Handle organization switch
  const handleOrgSwitch = (org: Organization) => {
    setActiveOrg(org)
    localStorage.setItem('org_slug', org.slug)
    
    // Navigate to the same page but with the new organization slug
    const currentPath = window.location.pathname
    
    // Check if the current path matches the pattern /:orgSlug/batches
    if (currentPath.match(/\/[^/]+\/batches/)) {
      navigate({ to: `/${org.slug}/batches` })
    } else {
      // For other paths, just replace the org slug
      const newPath = currentPath.replace(/\/[^/]+\//, `/${org.slug}/`)
      navigate({ to: newPath })
    }
  }

  // Handle create organization
  const handleCreateOrg = async () => {
    if (!newOrgName || !newOrgSlug) return
    
    setIsCreatingOrg(true)
    try {
      // In a real app, you would create the organization through the API
      const newOrg = await auth.createOrganization(newOrgName, newOrgSlug)
      
      setOrganizations([...organizations, newOrg])
      setActiveOrg(newOrg)
      localStorage.setItem('org_slug', newOrg.slug)
      
      setIsCreateOrgOpen(false)
      setNewOrgName('')
      setNewOrgSlug('')
      
      toast({
        title: 'Success',
        description: 'Organization created successfully',
      })
      
      // Navigate to the same page but with the new organization slug
      const currentPath = window.location.pathname
      
      // Check if the current path matches the pattern /:orgSlug/batches
      if (currentPath.match(/\/[^/]+\/batches/)) {
        navigate({ to: `/${newOrg.slug}/batches` })
      } else {
        // For other paths, just replace the org slug
        const newPath = currentPath.replace(/\/[^/]+\//, `/${newOrg.slug}/`)
        navigate({ to: newPath })
      }
    } catch (error) {
      console.error('Error creating organization:', error)
      toast({
        title: 'Error',
        description: 'Failed to create organization',
        variant: 'destructive',
      })
    } finally {
      setIsCreatingOrg(false)
    }
  }

  // Generate slug from name
  const generateSlug = (name: string) => {
    return name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  }

  // Handle name change and auto-generate slug
  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const name = e.target.value
    setNewOrgName(name)
    setNewOrgSlug(generateSlug(name))
  }

  if (isLoading || !activeOrg) {
    return (
      <SidebarMenu>
        <SidebarMenuItem>
          <SidebarMenuButton size='lg'>
            <div className='flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground'>
              <Building className='size-4' />
            </div>
            <div className='grid flex-1 text-left text-sm leading-tight'>
              <span className='truncate font-semibold'>Loading...</span>
              <span className='truncate text-xs'>Please wait</span>
            </div>
          </SidebarMenuButton>
        </SidebarMenuItem>
      </SidebarMenu>
    )
  }

  return (
    <>
      <SidebarMenu>
        <SidebarMenuItem>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <SidebarMenuButton
                size='lg'
                className='data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground'
              >
                <div className='flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground'>
                  <Building className='size-4' />
                </div>
                <div className='grid flex-1 text-left text-sm leading-tight'>
                  <span className='truncate font-semibold'>
                    {activeOrg.name}
                  </span>
                  <span className='truncate text-xs'>{activeOrg.slug}</span>
                </div>
                <ChevronsUpDown className='ml-auto' />
              </SidebarMenuButton>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              className='w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg'
              align='start'
              side={isMobile ? 'bottom' : 'right'}
              sideOffset={4}
            >
              <DropdownMenuLabel className='text-xs text-muted-foreground'>
                Organizations
              </DropdownMenuLabel>
              {organizations.map((org, index) => (
                <DropdownMenuItem
                  key={org.id}
                  onClick={() => handleOrgSwitch(org)}
                  className='gap-2 p-2'
                >
                  <div className='flex size-6 items-center justify-center rounded-sm border'>
                    <Building className='size-4 shrink-0' />
                  </div>
                  {org.name}
                  <DropdownMenuShortcut>⌘{index + 1}</DropdownMenuShortcut>
                </DropdownMenuItem>
              ))}
              <DropdownMenuSeparator />
              <DropdownMenuItem 
                className='gap-2 p-2'
                onClick={() => setIsCreateOrgOpen(true)}
              >
                <div className='flex size-6 items-center justify-center rounded-md border bg-background'>
                  <Plus className='size-4' />
                </div>
                <div className='font-medium text-muted-foreground'>Add organization</div>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </SidebarMenuItem>
      </SidebarMenu>

      {/* Create Organization Dialog */}
      <Dialog open={isCreateOrgOpen} onOpenChange={setIsCreateOrgOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Organization</DialogTitle>
            <DialogDescription>
              Create a new organization to manage your batches and files.
            </DialogDescription>
          </DialogHeader>
          <div className='grid gap-4 py-4'>
            <div className='grid gap-2'>
              <Label htmlFor='name'>Name</Label>
              <Input
                id='name'
                value={newOrgName}
                onChange={handleNameChange}
                placeholder='Enter organization name'
              />
            </div>
            <div className='grid gap-2'>
              <Label htmlFor='slug'>Slug</Label>
              <Input
                id='slug'
                value={newOrgSlug}
                onChange={(e) => setNewOrgSlug(e.target.value)}
                placeholder='Enter organization slug'
              />
              <p className='text-xs text-muted-foreground'>
                The slug will be used in URLs and must be unique.
              </p>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant='outline'
              onClick={() => setIsCreateOrgOpen(false)}
              disabled={isCreatingOrg}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateOrg}
              disabled={!newOrgName || !newOrgSlug || isCreatingOrg}
            >
              {isCreatingOrg ? 'Creating...' : 'Create'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
