import { useState, useEffect, useCallback } from 'react'
import { useToast } from '@/hooks/use-toast'
import { Button } from '@/components/ui/button'
import { ArrowLeftIcon } from 'lucide-react'
import { Project } from '@/types/project'
import { projectService } from '@/services/projectService'
import { Skeleton } from '@/components/ui/skeleton'
import { useNavigate, useParams } from '@tanstack/react-router'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { IconDatabase, IconMapPin } from '@tabler/icons-react'

export default function ProjectConfig() {
  const { toast } = useToast()
  const navigate = useNavigate()
  // Get organization slug and project name from route params
  const { orgSlug, projectName } = useParams({ 
    from: '/_authenticated/$orgSlug/$projectName/config' 
  })

  // State for project
  const [project, setProject] = useState<Project | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  // Fetch project
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
  
  // Fetch project on component mount
  useEffect(() => {
    fetchProject()
  }, [fetchProject])

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
      {/* Back button - only visible on mobile */}
      <div className="mb-6 flex items-center md:hidden">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate({ 
            to: '/$orgSlug/$projectName', 
            params: { orgSlug, projectName } 
          })}
          className="mr-4"
        >
          <ArrowLeftIcon className="mr-2 h-4 w-4" />
          Back to Project
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
              <h1 className="text-3xl font-bold tracking-tight">{project.name} - Configuration</h1>
              <p className="text-muted-foreground">
                Manage configuration settings for this project
              </p>
            </div>

            <Separator />

            <Card>
              <CardHeader>
                <CardTitle>Project Configuration</CardTitle>
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
              Back to Projects
            </Button>
          </div>
        )}
    </>
  )
}
