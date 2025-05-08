import { useState, useEffect, useCallback } from 'react'
import { useToast } from '@/hooks/use-toast'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { Button } from '@/components/ui/button'
import { PlusIcon, DatabaseIcon } from 'lucide-react'
import { Project, ProjectCreate } from '@/types/project'
import { projectService } from '@/services/projectService'
import { ProjectCard } from '@/components/project/project-card'
import { CreateProjectModal } from '@/components/project/create-project-modal'
import { DeleteProjectDialog } from '@/components/project/delete-project-dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { useNavigate, useParams } from '@tanstack/react-router'
import {
  IconAdjustmentsHorizontal,
  IconSortAscendingLetters,
  IconSortDescendingLetters,
} from '@tabler/icons-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'

export default function Projects() {
  const { toast } = useToast()
  const navigate = useNavigate()
  // Get organization slug from route params
  const { orgSlug } = useParams({ from: '/_authenticated/$orgSlug/projects' })

  // State for projects
  const [projects, setProjects] = useState<Project[]>([])
  const [isLoadingProjects, setIsLoadingProjects] = useState(true)
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)

  // State for modals
  const [isCreateProjectModalOpen, setIsCreateProjectModalOpen] = useState(false)
  const [isDeleteProjectModalOpen, setIsDeleteProjectModalOpen] = useState(false)

  // State for loading states
  const [isCreatingProject, setIsCreatingProject] = useState(false)
  const [isDeletingProject, setIsDeletingProject] = useState(false)

  // State for filtering and sorting
  const [sort, setSort] = useState('ascending')
  const [regionFilter, setRegionFilter] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')

  // Fetch projects
  const fetchProjects = useCallback(async () => {
    setIsLoadingProjects(true)
    try {
      const data = await projectService.getProjects(orgSlug)
      setProjects(data)
    } catch (error) {
      console.error('Error fetching projects:', error)
      toast({
        title: 'Error',
        description: 'Failed to fetch projects',
        variant: 'destructive',
      })
    } finally {
      setIsLoadingProjects(false)
    }
  }, [orgSlug, toast])
  
  // Fetch projects on component mount or when orgSlug changes
  useEffect(() => {
    fetchProjects()
  }, [fetchProjects])

  // Create project
  const handleCreateProject = async (project: ProjectCreate) => {
    setIsCreatingProject(true)
    try {
      const newProject = await projectService.createProject(project, orgSlug)
      setProjects((prev) => [...prev, newProject])
      setIsCreateProjectModalOpen(false)
      toast({
        title: 'Success',
        description: 'Project created successfully',
      })
    } catch (error) {
      console.error('Error creating project:', error)
      toast({
        title: 'Error',
        description: 'Failed to create project',
        variant: 'destructive',
      })
    } finally {
      setIsCreatingProject(false)
    }
  }

  // Delete project
  const handleDeleteProject = async () => {
    if (!selectedProject) return

    setIsDeletingProject(true)
    try {
      await projectService.deleteProject(selectedProject.name, orgSlug)
      setProjects((prev) => prev.filter((project) => project.id !== selectedProject.id))
      setIsDeleteProjectModalOpen(false)
      toast({
        title: 'Success',
        description: 'Project deleted successfully',
      })
    } catch (error) {
      console.error('Error deleting project:', error)
      toast({
        title: 'Error',
        description: 'Failed to delete project',
        variant: 'destructive',
      })
    } finally {
      setIsDeletingProject(false)
    }
  }

  // Filter and sort projects
  const filteredProjects = projects
    .filter((project) => {
      // Filter by region
      if (regionFilter !== 'all') {
        return project.data_region === regionFilter
      }
      return true
    })
    .filter((project) => {
      // Filter by search term
      return project.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (project.description && project.description.toLowerCase().includes(searchTerm.toLowerCase()))
    })
    .sort((a, b) => {
      // Sort by name
      return sort === 'ascending'
        ? a.name.localeCompare(b.name)
        : b.name.localeCompare(a.name)
    })

  // Get region display name
  const getRegionDisplay = (region: string) => {
    switch (region) {
      case 'all':
        return 'All Regions'
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
        <div className="mb-6 flex flex-wrap items-center justify-between gap-x-4 space-y-2">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Projects</h2>
            <p className="text-muted-foreground">
              Manage your projects and data storage locations
            </p>
          </div>
          <Button
            onClick={() => setIsCreateProjectModalOpen(true)}
            className="flex items-center"
          >
            <PlusIcon className="mr-2 h-4 w-4" />
            Create Project
          </Button>
        </div>

        <div className="my-4 flex items-end justify-between sm:my-0 sm:items-center">
          <div className="flex flex-col gap-4 sm:my-4 sm:flex-row">
            <Input
              placeholder="Search projects..."
              className="h-9 w-40 lg:w-[250px]"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            <Select value={regionFilter} onValueChange={setRegionFilter}>
              <SelectTrigger className="w-36">
                <SelectValue>{getRegionDisplay(regionFilter)}</SelectValue>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Regions</SelectItem>
                <SelectItem value="local">Local Storage</SelectItem>
                <SelectItem value="india">India</SelectItem>
                <SelectItem value="us">United States</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Select value={sort} onValueChange={setSort}>
            <SelectTrigger className="w-16">
              <SelectValue>
                <IconAdjustmentsHorizontal size={18} />
              </SelectValue>
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value="ascending">
                <div className="flex items-center gap-4">
                  <IconSortAscendingLetters size={16} />
                  <span>Ascending</span>
                </div>
              </SelectItem>
              <SelectItem value="descending">
                <div className="flex items-center gap-4">
                  <IconSortDescendingLetters size={16} />
                  <span>Descending</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator className="shadow" />

        {isLoadingProjects ? (
          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-[200px] w-full" />
            ))}
          </div>
        ) : filteredProjects.length === 0 ? (
          <div className="flex h-[300px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
            <DatabaseIcon className="mb-4 h-12 w-12 text-muted-foreground" />
            <h3 className="mb-2 text-lg font-medium">No projects found</h3>
            <p className="mb-4 text-sm text-muted-foreground">
              Create your first project to start organizing your data
            </p>
            <Button
              onClick={() => setIsCreateProjectModalOpen(true)}
              className="flex items-center"
            >
              <PlusIcon className="mr-2 h-4 w-4" />
              Create Project
            </Button>
          </div>
        ) : (
          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onViewProject={(project) => {
                  navigate({
                    to: '/$orgSlug/$projectName',
                    params: {
                      orgSlug,
                      projectName: project.name
                    }
                  })
                }}
                onEditProject={(project) => {
                  // In a real app, you would open an edit modal
                  toast({
                    title: 'Edit Project',
                    description: `Editing project: ${project.name}`,
                  })
                }}
                onDeleteProject={(project) => {
                  setSelectedProject(project)
                  setIsDeleteProjectModalOpen(true)
                }}
              />
            ))}
          </div>
        )}
      </Main>

      {/* Modals */}
      <CreateProjectModal
        isOpen={isCreateProjectModalOpen}
        onClose={() => setIsCreateProjectModalOpen(false)}
        onCreateProject={handleCreateProject}
        isLoading={isCreatingProject}
      />

      <DeleteProjectDialog
        isOpen={isDeleteProjectModalOpen}
        onClose={() => setIsDeleteProjectModalOpen(false)}
        onConfirm={handleDeleteProject}
        title="Delete Project"
        description={`Are you sure you want to delete the project "${selectedProject?.name}"? This action cannot be undone and all associated data will be deleted.`}
        isLoading={isDeletingProject}
      />
    </>
  )
}
