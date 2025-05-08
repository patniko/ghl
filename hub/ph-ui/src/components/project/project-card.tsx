import { IconDotsVertical, IconEdit, IconTrash } from '@tabler/icons-react'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Project } from '@/types/project'
import { formatDistanceToNow } from 'date-fns'
import { IconDatabase, IconMapPin } from '@tabler/icons-react'

interface ProjectCardProps {
  project: Project
  onViewProject: (project: Project) => void
  onEditProject: (project: Project) => void
  onDeleteProject: (project: Project) => void
}

export function ProjectCard({
  project,
  onViewProject,
  onEditProject,
  onDeleteProject,
}: ProjectCardProps) {
  // Format the created_at date
  const formattedDate = formatDistanceToNow(new Date(project.created_at), {
    addSuffix: true,
  })

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
    <Card className="overflow-hidden transition-all hover:shadow-md">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <div className="space-y-1">
            <CardTitle className="text-xl">{project.name}</CardTitle>
            <CardDescription>Created {formattedDate}</CardDescription>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="h-8 w-8">
                <IconDotsVertical className="h-4 w-4" />
                <span className="sr-only">Open menu</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onEditProject(project)}>
                <IconEdit className="mr-2 h-4 w-4" />
                Edit
              </DropdownMenuItem>
              <DropdownMenuItem
                onClick={() => onDeleteProject(project)}
                className="text-destructive focus:text-destructive"
              >
                <IconTrash className="mr-2 h-4 w-4" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent>
        <div className="text-sm text-muted-foreground">
          {project.description || 'No description provided'}
        </div>
        <div className="mt-4 flex items-center text-xs text-muted-foreground">
          {getRegionIcon(project.data_region)}
          <span className="ml-1">{getRegionDisplay(project.data_region)}</span>
        </div>
      </CardContent>
      <CardFooter className="border-t bg-muted/50 px-6 py-3">
        <Button
          variant="ghost"
          size="sm"
          className="w-full"
          onClick={() => onViewProject(project)}
        >
          View Project
        </Button>
      </CardFooter>
    </Card>
  )
}
