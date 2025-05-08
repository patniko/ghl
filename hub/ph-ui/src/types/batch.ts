export interface Batch {
  id: number
  name: string
  description?: string
  created_at: string
  updated_at: string
  organization_id: number
  user_id: number
  status: string
  quality_summary?: QualitySummary
  project_id?: number
}

export interface BatchCreate {
  name: string
  description?: string
  project_id?: number
}

export interface BatchUpdate {
  name?: string
  description?: string
  status?: string
}

export interface BatchFile {
  id: number
  batch_id: number
  original_filename: string
  file_type: string
  file_size: number
  created_at: string
  updated_at: string
  processing_status: string
  schema_type?: string
  csv_headers?: string[]
  csv_rows_count?: number
  metadata?: Record<string, unknown>
}

export interface BatchStatistics {
  total_batches: number
  total_files: number
  files_by_type: Record<string, number>
  files_by_status: Record<string, number>
}

export interface QualitySummary {
  total_checks: number
  passed_checks: number
  failed_checks: number
  issues_by_severity: {
    error: number
    warning: number
    info: number
  }
}
