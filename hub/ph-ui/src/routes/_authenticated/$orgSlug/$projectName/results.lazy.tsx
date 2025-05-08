import { createLazyFileRoute } from '@tanstack/react-router'
import ProjectResults from '@/features/projects/project-results'

export const Route = createLazyFileRoute(
  '/_authenticated/$orgSlug/$projectName/results',
)({
  component: ProjectResults,
})
