import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarRail,
} from '@/components/ui/sidebar'
import { NavGroup } from '@/components/layout/nav-group'
import { TeamSwitcher } from '@/components/layout/team-switcher'
import { sidebarData } from './data/sidebar-data'
import { projectSidebarData } from './data/project-sidebar-data'
import { useLocation } from '@tanstack/react-router'
import { useEffect, useState } from 'react'
import { projectService } from '@/services/projectService'
import { Project } from '@/types/project'
import { IconDatabase } from '@tabler/icons-react'
import { NavLink } from './types'

export function AppSidebar({ ...props }: React.ComponentProps<typeof Sidebar>) {
  const location = useLocation()
  const [isProjectContext, setIsProjectContext] = useState(false)
  const [recentProjects, setRecentProjects] = useState<Project[]>([])
  const [currentSidebarData, setCurrentSidebarData] = useState(sidebarData)
  
  // Check if we're in a project context based on the URL
  useEffect(() => {
    const path = location.pathname
    
    // Extract path segments
    const segments = path.split('/').filter(segment => segment !== '')
    
    // Project routes have the pattern /$orgSlug/$projectName
    // But we need to exclude specific routes like settings, help-center, etc.
    const isProject = segments.length >= 2 && 
                      !['settings', 'help-center', 'apps', 'chats', 
                        'tasks', 'users', 'projects'].includes(segments[0]) &&
                      !['projects'].includes(segments[1])
    
    setIsProjectContext(isProject)
  }, [location.pathname])
  
  // Fetch recent projects
  useEffect(() => {
    const fetchRecentProjects = async () => {
      try {
        const orgSlug = localStorage.getItem('org_slug') || 'personal'
        const projects = await projectService.getProjects(orgSlug)
        setRecentProjects(projects.slice(0, 3)) // Get the 3 most recent projects
      } catch (error) {
        console.error('Error fetching recent projects:', error)
      }
    }
    
    if (!isProjectContext) {
      fetchRecentProjects()
    }
  }, [isProjectContext])
  
  // Update sidebar data based on context and recent projects
  useEffect(() => {
    if (isProjectContext) {
      setCurrentSidebarData(projectSidebarData)
    } else {
      // Only show Recent section if there are recent projects
      if (recentProjects.length > 0) {
        const updatedSidebarData = { ...sidebarData }
        
        // Find the Recent section
        const recentGroupIndex = updatedSidebarData.navGroups.findIndex(
          group => group.title === 'Recent'
        )
        
        if (recentGroupIndex !== -1) {
          // Replace the items with actual recent projects
          updatedSidebarData.navGroups[recentGroupIndex].items = recentProjects.map(project => ({
            title: project.name,
            url: '/$orgSlug/projects', // Use a valid route
            icon: IconDatabase,
            dynamic: true
          } as NavLink))
          
          setCurrentSidebarData(updatedSidebarData)
        }
      } else {
        // If no recent projects, remove the Recent section
        const updatedSidebarData = { ...sidebarData }
        updatedSidebarData.navGroups = updatedSidebarData.navGroups.filter(
          group => group.title !== 'Recent'
        )
        
        setCurrentSidebarData(updatedSidebarData)
      }
    }
  }, [isProjectContext, recentProjects])

  return (
    <Sidebar collapsible='icon' variant='floating' {...props}>
      <SidebarHeader>
        <TeamSwitcher />
      </SidebarHeader>
      <SidebarContent>
        {currentSidebarData.navGroups.map((props) => (
          <NavGroup key={props.title} {...props} />
        ))}
      </SidebarContent>
  {/* SidebarFooter with NavUser removed to hide user profile section */}
      <SidebarRail />
    </Sidebar>
  )
}
