import { createFileRoute, redirect } from '@tanstack/react-router'
import Projects from '@/features/projects'

export const Route = createFileRoute('/_authenticated/')({
  component: Projects,
  beforeLoad: async () => {
    // Get the organization slug from localStorage or use a default
    const orgSlug = localStorage.getItem('org_slug') || 'personal'
    
    // Redirect to the organization-specific projects page
    throw redirect({
      to: '/$orgSlug',
      params: { orgSlug },
    })
  }
})
