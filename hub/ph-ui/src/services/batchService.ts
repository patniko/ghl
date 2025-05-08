import api from './api'
import { Batch, BatchCreate, BatchFile, BatchStatistics, BatchUpdate, QualitySummary } from '../types/batch'

export interface AllFilesParams {
  project_id?: number
  file_type?: string
  search?: string
  sort_by?: 'created_at' | 'filename' | 'file_size' | 'file_type'
  sort_order?: 'asc' | 'desc'
  page?: number
  page_size?: number
}

export interface AllFilesResponse {
  total: number
  page: number
  page_size: number
  total_pages: number
  files: (BatchFile & {
    batch_name: string | null
    project_name: string | null
  })[]
}

export const batchService = {
  // Create a new batch
  createBatch: async (batch: BatchCreate, orgSlug?: string): Promise<Batch> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}` : '/batches'
    const response = await api.post(endpoint, batch)
    return response.data
  },
  
  // Get all batches
  getBatches: async (orgSlug?: string): Promise<Batch[]> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}` : '/batches'
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get all files across all batches with pagination, search, and sorting
  getAllFiles: async (params: AllFilesParams = {}, orgSlug?: string): Promise<AllFilesResponse> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}/all-files` : '/batches/all-files'
    const response = await api.get(endpoint, { params })
    return response.data
  },
  
  // Get a specific batch by ID
  getBatch: async (id: number, orgSlug?: string): Promise<Batch> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}/${id}` : `/batches/${id}`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Update a batch
  updateBatch: async (id: number, update: BatchUpdate, orgSlug?: string): Promise<Batch> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}/${id}` : `/batches/${id}`
    const response = await api.put(endpoint, update)
    return response.data
  },
  
  // Delete a batch
  deleteBatch: async (id: number, orgSlug?: string): Promise<void> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}/${id}` : `/batches/${id}`
    await api.delete(endpoint)
  },
  
  // Get batch statistics
  getBatchStatistics: async (orgSlug?: string): Promise<BatchStatistics> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}/statistics` : '/batches/statistics'
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get batch quality summary
  getBatchQualitySummary: async (batchId: number, orgSlug?: string): Promise<QualitySummary> => {
    const endpoint = orgSlug ? `/batches/${orgSlug}/${batchId}/quality-summary` : `/batches/${batchId}/quality-summary`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get batch files
  getBatchFiles: async (batchId: number, fileType?: string, orgSlug?: string): Promise<BatchFile[]> => {
    const params: Record<string, string> = {}
    if (fileType) params.file_type = fileType
    
    const endpoint = orgSlug ? `/batches/${orgSlug}/${batchId}/files` : `/batches/${batchId}/files`
    const response = await api.get(endpoint, { params })
    return response.data
  }
}

export const fileService = {
  // Upload a file to a batch
  uploadFile: async (file: File, batchId: number, fileType?: string, orgSlug?: string): Promise<BatchFile> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('batch_id', batchId.toString())
    
    if (fileType) {
      formData.append('file_type', fileType)
    }
    
    const endpoint = orgSlug ? `/${orgSlug}/files/upload` : '/files/upload'
    const response = await api.post(
      endpoint,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )
    return response.data
  },
  
  // Get all files
  getFiles: async (batchId?: number, fileType?: string, orgSlug?: string): Promise<BatchFile[]> => {
    const params: Record<string, string> = {}
    if (batchId) params.batch_id = batchId.toString()
    if (fileType) params.file_type = fileType
    
    const endpoint = orgSlug ? `/${orgSlug}/files/` : '/files/'
    const response = await api.get(endpoint, { params })
    return response.data
  },
  
  // Get a specific file
  getFile: async (id: number, orgSlug?: string): Promise<BatchFile> => {
    const endpoint = orgSlug ? `/${orgSlug}/files/${id}` : `/files/${id}`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Delete a file
  deleteFile: async (id: number, orgSlug?: string): Promise<void> => {
    const endpoint = orgSlug ? `/${orgSlug}/files/${id}` : `/files/${id}`
    await api.delete(endpoint)
  },
  
  // Get file download URL
  getFileDownloadUrl: (fileId: number, orgSlug?: string): string => {
    const endpoint = orgSlug ? `/${orgSlug}/files/${fileId}/download` : `/files/${fileId}/download`
    return `${api.defaults.baseURL}${endpoint}`
  },
  
  // Get CSV preview
  getCsvPreview: async (fileId: number, orgSlug?: string): Promise<{
    headers: string[]
    data: string[][]
    total_rows: number
    preview_rows: number
  }> => {
    const endpoint = orgSlug ? `/${orgSlug}/files/previews/csv/${fileId}` : `/files/previews/csv/${fileId}`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get potential column mappings for a CSV file
  getPotentialMappings: async (fileId: number, orgSlug?: string): Promise<Record<string, string[]>> => {
    const endpoint = orgSlug ? `/${orgSlug}/files/${fileId}/potential-mappings` : `/files/${fileId}/potential-mappings`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Update potential column mappings for a CSV file
  updatePotentialMappings: async (fileId: number, mappings: Record<string, string[]>, orgSlug?: string): Promise<Record<string, string[]>> => {
    const endpoint = orgSlug ? `/${orgSlug}/files/${fileId}/potential-mappings` : `/files/${fileId}/potential-mappings`
    const response = await api.put(endpoint, mappings)
    return response.data
  },
  
  // Apply checks to a CSV file
  applyChecks: async (fileId: number, columnChecks: Record<string, number[]>, orgSlug?: string): Promise<Record<string, unknown>> => {
    const endpoint = orgSlug ? `/${orgSlug}/files/${fileId}/apply-checks` : `/files/${fileId}/apply-checks`
    const response = await api.post(endpoint, columnChecks)
    return response.data
  }
}
