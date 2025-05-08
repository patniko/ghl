export interface Project {
  id: number
  name: string
  description?: string
  data_region: string
  s3_bucket_name?: string
  created_at: string
  updated_at: string
  organization_id: number
  user_id: number
}

export interface ProjectCreate {
  name: string
  description?: string
  data_region: string
  s3_bucket_name?: string
}

export interface ProjectUpdate {
  name?: string
  description?: string
  data_region?: string
  s3_bucket_name?: string
}

export interface ProjectStatistics {
  total_projects: number
  projects_by_region: Record<string, number>
}

export enum DataRegion {
  LOCAL = "local",
  INDIA = "india",
  US = "us"
}

export interface DataRegionInfo {
  id: DataRegion
  name: string
  description: string
  icon: React.ReactNode
  available: boolean
}
