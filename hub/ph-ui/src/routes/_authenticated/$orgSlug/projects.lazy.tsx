import { createLazyFileRoute } from '@tanstack/react-router'
import Projects from '@/features/projects'

export const Route = createLazyFileRoute('/_authenticated/$orgSlug/projects')({
  component: Projects,
})
