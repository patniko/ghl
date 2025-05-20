export interface SyntheticDataset {
  id: number
  user_id: number
  organization_id?: number
  batch_id: number | null
  name: string
  description: string | null
  num_patients: number
  column_mappings: Record<string, unknown> | null
  applied_checks: Record<string, unknown> | null
  check_results: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface SyntheticDatasetCreate {
  name: string
  description?: string
  num_patients: number
  batch_id?: number
}

export interface SamplesDatasetCreate extends SyntheticDatasetCreate {
  data_types: string[]
  include_partials: boolean
  partial_rate: number
}

export interface SyntheticDatasetUpdate {
  name?: string
  description?: string
  batch_id?: number
  column_mappings?: Record<string, unknown>
  applied_checks?: Record<string, unknown>
}

export interface SyntheticDatasetData {
  data: Record<string, unknown>[]
  pagination: {
    page: number
    page_size: number
    total_items: number
    total_pages: number
  }
}

export interface SyntheticDatasetProgress {
  dataset_id: number
  progress: number
  completed: boolean
  current: number
  total: number
}

export interface ColumnTypeResult {
  dataset_id: number
  dataset_name: string
  column_types: Record<string, string>
}
