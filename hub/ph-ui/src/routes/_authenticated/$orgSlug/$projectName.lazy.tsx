import { createLazyFileRoute } from '@tanstack/react-router'
import ProjectLayout from '@/features/projects/project-layout'

export const Route = createLazyFileRoute(
  '/_authenticated/$orgSlug/$projectName',
)({
  component: ProjectLayout,
})
