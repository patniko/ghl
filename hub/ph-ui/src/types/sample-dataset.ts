export interface SampleDataset {
  id: number
  user_id: number
  organization_id: number
  batch_id: number | null
  name: string
  description: string | null
  num_patients: number
  created_at: string
  updated_at: string
}

export interface SamplesDatasetCreate {
  name: string
  description?: string
  num_patients: number
  batch_id?: number
  data_types: string[]
  include_partials: boolean
  partial_rate: number
}

export interface SamplesDatasetUpdate {
  name?: string
  description?: string
  batch_id?: number
}

export interface SamplesDatasetData {
  data: Record<string, unknown>[]
  pagination: {
    page: number
    page_size: number
    total_items: number
    total_pages: number
  }
}

export interface SamplesDatasetProgress {
  dataset_id: number
  progress: number
  completed: boolean
  current: number
  total: number
}
