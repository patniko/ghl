import api from './api'
import { 
  SyntheticDataset, 
  SyntheticDatasetCreate, 
  SyntheticDatasetUpdate, 
  SyntheticDatasetData,
  ColumnTypeResult,
  SamplesDatasetCreate
} from '../types/synthetic-dataset'

export const syntheticDatasetsService = {
  // Create a new synthetic dataset
  createDataset: async (dataset: SyntheticDatasetCreate, orgSlug: string): Promise<SyntheticDataset> => {
    const endpoint = `/${orgSlug}/synthetic-datasets`
    const response = await api.post(endpoint, dataset)
    return response.data
  },
  
  // Get all synthetic datasets
  getDatasets: async (orgSlug: string): Promise<SyntheticDataset[]> => {
    const endpoint = `/${orgSlug}/synthetic-datasets`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get a specific synthetic dataset by ID
  getDataset: async (datasetId: number, orgSlug: string): Promise<SyntheticDataset> => {
    const endpoint = `/${orgSlug}/synthetic-datasets/${datasetId}`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get data from a synthetic dataset with pagination
  getDatasetData: async (
    datasetId: number, 
    orgSlug: string, 
    page: number = 1, 
    pageSize: number = 100
  ): Promise<SyntheticDatasetData> => {
    const endpoint = `/${orgSlug}/synthetic-datasets/${datasetId}/data`
    const response = await api.get(endpoint, {
      params: { page, page_size: pageSize }
    })
    return response.data
  },
  
  // Update a synthetic dataset
  updateDataset: async (
    datasetId: number, 
    update: SyntheticDatasetUpdate, 
    orgSlug: string
  ): Promise<SyntheticDataset> => {
    const endpoint = `/${orgSlug}/synthetic-datasets/${datasetId}`
    const response = await api.put(endpoint, update)
    return response.data
  },
  
  // Delete a synthetic dataset
  deleteDataset: async (datasetId: number, orgSlug: string): Promise<void> => {
    const endpoint = `/${orgSlug}/synthetic-datasets/${datasetId}`
    await api.delete(endpoint)
  },
  
  // Upload a CSV file to replace dataset data
  uploadCsv: async (datasetId: number, file: File, orgSlug: string): Promise<{ 
    message: string, 
    num_records: number, 
    columns: string[] 
  }> => {
    const endpoint = `/${orgSlug}/synthetic-datasets/${datasetId}/upload-csv`
    
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    return response.data
  },
  
  // Infer column types for a dataset
  inferColumnTypes: async (datasetId: number, orgSlug: string): Promise<ColumnTypeResult> => {
    const endpoint = `/${orgSlug}/synthetic-datasets/${datasetId}/infer-column-types`
    const response = await api.post(endpoint)
    return response.data
  },
  
  // Get WebSocket URL for progress updates
  getWebSocketUrl: (userId: number): string => {
    const baseUrl = api.defaults.baseURL || ''
    // Convert http/https to ws/wss
    const wsBaseUrl = baseUrl.replace(/^http/, 'ws')
    return `${wsBaseUrl}/synthetic-datasets/ws/progress?user_id=${userId}`
  },
  
  // Samples Datasets API
  
  // Create a new samples dataset
  createSamplesDataset: async (
    dataset: SamplesDatasetCreate, 
    orgSlug: string
  ): Promise<SyntheticDataset> => {
    const endpoint = `/${orgSlug}/samples-datasets`
    
    // Convert to FormData
    const formData = new FormData()
    formData.append('name', dataset.name)
    if (dataset.description) {
      formData.append('description', dataset.description)
    }
    formData.append('num_patients', dataset.num_patients.toString())
    
    // Add data types
    if (dataset.data_types && dataset.data_types.length > 0) {
      dataset.data_types.forEach((type: string) => {
        formData.append('data_types', type)
      })
    }
    
    // Add other options
    formData.append('include_partials', dataset.include_partials ? 'true' : 'false')
    formData.append('partial_rate', dataset.partial_rate.toString())
    
    if (dataset.batch_id) {
      formData.append('batch_id', dataset.batch_id.toString())
    }
    
    const response = await api.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    return response.data
  },
  
  // Get all samples datasets
  getSamplesDatasets: async (orgSlug: string): Promise<SyntheticDataset[]> => {
    const endpoint = `/${orgSlug}/samples-datasets`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get a specific samples dataset by ID
  getSamplesDataset: async (datasetId: number, orgSlug: string): Promise<SyntheticDataset> => {
    const endpoint = `/${orgSlug}/samples-datasets/${datasetId}`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Delete a samples dataset
  deleteSamplesDataset: async (datasetId: number, orgSlug: string): Promise<void> => {
    const endpoint = `/${orgSlug}/samples-datasets/${datasetId}`
    await api.delete(endpoint)
  },
  
  // Download a samples dataset
  downloadSamplesDataset: async (datasetId: number, orgSlug: string): Promise<Blob> => {
    const endpoint = `/${orgSlug}/samples-datasets/${datasetId}/download`
    const response = await api.get(endpoint, {
      responseType: 'blob'
    })
    return response.data
  }
}
