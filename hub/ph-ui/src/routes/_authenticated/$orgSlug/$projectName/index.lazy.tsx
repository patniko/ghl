import { createLazyFileRoute } from '@tanstack/react-router'
import ProjectDetails from '@/features/projects/project-details'

export const Route = createLazyFileRoute(
  '/_authenticated/$orgSlug/$projectName/',
)({
  component: ProjectDetails,
})
