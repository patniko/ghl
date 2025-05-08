import api from './api'
import { Project, ProjectCreate, ProjectStatistics, ProjectUpdate } from '../types/project'

export const projectService = {
  // Create a new project
  createProject: async (project: ProjectCreate, orgSlug?: string): Promise<Project> => {
    const endpoint = orgSlug ? `/projects/${orgSlug}` : '/projects'
    const response = await api.post(endpoint, project)
    return response.data
  },
  
  // Get all projects
  getProjects: async (orgSlug?: string): Promise<Project[]> => {
    const endpoint = orgSlug ? `/projects/${orgSlug}` : '/projects'
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get a specific project by ID (deprecated)
  getProject: async (id: number, orgSlug?: string): Promise<Project> => {
    // First get all projects
    const projects = await projectService.getProjects(orgSlug)
    // Find the project with the matching ID
    const project = projects.find(p => p.id === id)
    
    if (!project) {
      throw new Error(`Project with ID "${id}" not found`)
    }
    
    return project
  },
  
  // Get a specific project by name
  getProjectByName: async (name: string, orgSlug?: string): Promise<Project> => {
    const endpoint = orgSlug ? `/projects/${orgSlug}/${name}` : `/projects/${name}`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Update a project
  updateProject: async (name: string, update: ProjectUpdate, orgSlug?: string): Promise<Project> => {
    const endpoint = orgSlug ? `/projects/${orgSlug}/${name}` : `/projects/${name}`
    const response = await api.put(endpoint, update)
    return response.data
  },
  
  // Delete a project
  deleteProject: async (name: string, orgSlug?: string): Promise<void> => {
    const endpoint = orgSlug ? `/projects/${orgSlug}/${name}` : `/projects/${name}`
    await api.delete(endpoint)
  },
  
  // Get project statistics
  getProjectStatistics: async (orgSlug?: string): Promise<ProjectStatistics> => {
    const endpoint = orgSlug ? `/projects/${orgSlug}/statistics` : '/projects/statistics'
    const response = await api.get(endpoint)
    return response.data
  }
}
