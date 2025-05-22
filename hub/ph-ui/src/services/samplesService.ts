import api from './api'
import { 
  SampleDataset, 
  SamplesDatasetCreate 
} from '../types/sample-dataset'

export const samplesService = {
  // Create a new samples dataset
  createSamplesDataset: async (
    dataset: SamplesDatasetCreate, 
    orgSlug: string
  ): Promise<SampleDataset> => {
    const endpoint = `/${orgSlug}/samples`
    
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
  getSamplesDatasets: async (orgSlug: string): Promise<SampleDataset[]> => {
    const endpoint = `/${orgSlug}/samples`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Get a specific samples dataset by ID
  getSamplesDataset: async (datasetId: number, orgSlug: string): Promise<SampleDataset> => {
    const endpoint = `/${orgSlug}/samples/${datasetId}`
    const response = await api.get(endpoint)
    return response.data
  },
  
  // Delete a samples dataset
  deleteSamplesDataset: async (datasetId: number, orgSlug: string): Promise<void> => {
    const endpoint = `/${orgSlug}/samples/${datasetId}`
    await api.delete(endpoint)
  },
  
  // Download a samples dataset
  downloadSamplesDataset: async (datasetId: number, orgSlug: string): Promise<Blob> => {
    const endpoint = `/${orgSlug}/samples/${datasetId}/download`
    const response = await api.get(endpoint, {
      responseType: 'blob'
    })
    return response.data
  }
}
