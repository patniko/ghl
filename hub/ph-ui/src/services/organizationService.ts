import api from './api'

export interface Organization {
  id: number
  name: string
  slug: string
  description?: string
  created_at: string
  updated_at: string
}

export interface OrganizationCreate {
  name: string
  slug: string
  description?: string
}

export const organizationService = {
  // Create a new organization
  createOrganization: async (organization: OrganizationCreate, orgSlug?: string): Promise<Organization> => {
    console.log(`[DEBUG] Creating organization:`, organization);
    try {
      const endpoint = orgSlug ? `/orgs/${orgSlug}` : '/orgs';
      const response = await api.post(endpoint, organization);
      console.log(`[DEBUG] Organization created:`, response.data);
      return response.data;
    } catch (error) {
      console.error(`[DEBUG] Error creating organization:`, error);
      throw error;
    }
  },
  
  // Get all organizations
  getOrganizations: async (orgSlug?: string): Promise<Organization[]> => {
    try {
      const endpoint = orgSlug ? `/orgs/${orgSlug}` : '/orgs';
      const response = await api.get(endpoint);
      return response.data;
    } catch (error) {
      console.error(`[DEBUG] Error getting organizations:`, error);
      throw error;
    }
  },
  
  // Get my organizations
  getMyOrganizations: async (orgSlug?: string): Promise<Organization[]> => {
    try {
      const endpoint = orgSlug ? `/orgs/${orgSlug}/me` : '/orgs/me';
      const response = await api.get(endpoint);
      return response.data;
    } catch (error) {
      console.error(`[DEBUG] Error getting my organizations:`, error);
      throw error;
    }
  },
  
  // Get a specific organization by slug
  getOrganization: async (slug: string, orgSlug?: string): Promise<Organization> => {
    try {
      const endpoint = orgSlug ? `/orgs/${orgSlug}/${slug}` : `/orgs/${slug}`;
      const response = await api.get(endpoint);
      return response.data;
    } catch (error) {
      console.error(`[DEBUG] Error getting organization:`, error);
      throw error;
    }
  },
  
  // Update an organization
  updateOrganization: async (orgId: number, update: Partial<OrganizationCreate>, orgSlug?: string): Promise<Organization> => {
    try {
      const endpoint = orgSlug ? `/orgs/${orgSlug}/${orgId}` : `/orgs/${orgId}`;
      const response = await api.put(endpoint, update);
      return response.data;
    } catch (error) {
      console.error(`[DEBUG] Error updating organization:`, error);
      throw error;
    }
  },
  
  // Delete an organization
  deleteOrganization: async (orgId: number, orgSlug?: string): Promise<void> => {
    try {
      const endpoint = orgSlug ? `/orgs/${orgSlug}/${orgId}` : `/orgs/${orgId}`;
      await api.delete(endpoint);
    } catch (error) {
      console.error(`[DEBUG] Error deleting organization:`, error);
      throw error;
    }
  },
  
  // Reset system checks for an organization
  resetSystemChecks: async (orgId: number, orgSlug?: string): Promise<void> => {
    try {
      const endpoint = orgSlug ? `/orgs/${orgSlug}/${orgId}/reset-checks` : `/orgs/${orgId}/reset-checks`;
      await api.post(endpoint);
    } catch (error) {
      console.error(`[DEBUG] Error resetting system checks:`, error);
      throw error;
    }
  }
}
