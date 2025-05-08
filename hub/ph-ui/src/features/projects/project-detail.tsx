import { useState, useEffect, useCallback } from 'react'
import { useToast } from '@/hooks/use-toast'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { Button } from '@/components/ui/button'
import { ArrowLeftIcon } from 'lucide-react'
import { Project } from '@/types/project'
import { projectService } from '@/services/projectService'
import { Skeleton } from '@/components/ui/skeleton'
import { Outlet, useNavigate, useParams, useRouterState } from '@tanstack/react-router'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { IconDatabase, IconMapPin } from '@tabler/icons-react'

export default function ProjectDetail() {
  const { toast } = useToast()
  const navigate = useNavigate()
  const routerState = useRouterState()
  // Get organization slug and project ID from route params
  const { orgSlug, projectName } = useParams({ 
    from: '/_authenticated/$orgSlug/$projectName' 
  })
  
  // Check if we're on the exact project route or a child route
  const currentPath = routerState.location.pathname
  // Check if the current path ends with the project name and doesn't have additional segments
  const isExactProjectRoute = currentPath.endsWith(`/${orgSlug}/${projectName}`) && 
                             !currentPath.endsWith(`/${orgSlug}/${projectName}/batches`) &&
                             !currentPath.endsWith(`/${orgSlug}/${projectName}/config`) &&
                             !currentPath.endsWith(`/${orgSlug}/${projectName}/results`) &&
                             !currentPath.endsWith(`/${orgSlug}/${projectName}/tasks`)
  
  // State for project
  const [project, setProject] = useState<Project | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Debug routing information
  console.log('ProjectDetail - Current Path:', currentPath)
  console.log('ProjectDetail - Is Exact Project Route:', isExactProjectRoute)
  console.log('ProjectDetail - Router State:', routerState)
  
  // Fetch project function
  const fetchProject = useCallback(async () => {
    setIsLoading(true)
    try {
      const data = await projectService.getProjectByName(projectName, orgSlug)
      setProject(data)
    } catch (error) {
      console.error('Error fetching project:', error)
      toast({
        title: 'Error',
        description: 'Failed to fetch project details',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }, [orgSlug, projectName, toast])
  
  // Fetch project on mount if we're on the exact project route
  useEffect(() => {
    if (isExactProjectRoute) {
      fetchProject()
    }
  }, [isExactProjectRoute, fetchProject])
  
  // If we're not on the exact project route, don't render anything except the Outlet
  if (!isExactProjectRoute) {
    return <Outlet />
  }

  // Get region display name
  const getRegionDisplay = (region: string) => {
    switch (region) {
      case 'local':
        return 'Local Storage'
      case 'india':
        return 'India'
      case 'us':
        return 'United States'
      default:
        return region
    }
  }

  // Get region icon
  const getRegionIcon = (region: string) => {
    switch (region) {
      case 'local':
        return <IconDatabase className="h-4 w-4" />
      default:
        return <IconMapPin className="h-4 w-4" />
    }
  }

  return (
    <>
      <Header fixed>
        <Search />
        <div className="ml-auto flex items-center space-x-4">
          <ThemeSwitch />
          <ProfileDropdown />
        </div>
      </Header>

      <Main>
        <div className="mb-6 flex items-center">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate({ to: '/$orgSlug', params: { orgSlug } })}
            className="mr-4"
          >
            <ArrowLeftIcon className="mr-2 h-4 w-4" />
            Back to Dashboard
          </Button>
        </div>

        {isLoading ? (
          <div className="space-y-4">
            <Skeleton className="h-8 w-1/3" />
            <Skeleton className="h-4 w-1/2" />
            <Skeleton className="mt-6 h-[300px] w-full" />
          </div>
        ) : project ? (
          <div className="space-y-6">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">{project.name}</h1>
              <p className="text-muted-foreground">
                {project.description || 'No description provided'}
              </p>
            </div>

            <Separator />

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Project Details</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h3 className="font-medium">Data Region</h3>
                    <div className="mt-1 flex items-center text-sm text-muted-foreground">
                      {project.data_region && getRegionIcon(project.data_region)}
                      <span className="ml-1">
                        {project.data_region && getRegionDisplay(project.data_region)}
                      </span>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium">Storage</h3>
                    <p className="text-sm text-muted-foreground">
                      {project.s3_bucket_name || 'Default storage'}
                    </p>
                  </div>

                  <div>
                    <h3 className="font-medium">Created</h3>
                    <p className="text-sm text-muted-foreground">
                      {new Date(project.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Project Stats</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <h3 className="font-medium">Batches</h3>
                    <div className="flex items-center justify-between">
                      <p className="text-sm text-muted-foreground">0 batches</p>
                      <Button
                        variant="link"
                        size="sm"
                        className="h-auto p-0 text-xs"
                        onClick={() => navigate({ 
                          to: '/$orgSlug/$projectName/batches', 
                          params: { orgSlug, projectName } 
                        })}
                      >
                        View
                      </Button>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium">Files</h3>
                    <p className="text-sm text-muted-foreground">0 files</p>
                  </div>

                  <div>
                    <h3 className="font-medium">Last Activity</h3>
                    <p className="text-sm text-muted-foreground">
                      {new Date(project.updated_at).toLocaleDateString()}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          <div className="flex h-[300px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
            <h3 className="mb-2 text-lg font-medium">Project not found</h3>
            <p className="mb-4 text-sm text-muted-foreground">
              The project you're looking for doesn't exist or you don't have access to it.
            </p>
            <Button
              onClick={() => navigate({ to: '/$orgSlug', params: { orgSlug } })}
              className="flex items-center"
            >
              <ArrowLeftIcon className="mr-2 h-4 w-4" />
              Back to Dashboard
            </Button>
          </div>
        )}
      </Main>
    </>
  )
}
