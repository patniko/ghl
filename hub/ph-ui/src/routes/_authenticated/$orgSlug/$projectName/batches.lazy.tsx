import { createLazyFileRoute } from '@tanstack/react-router'
import ProjectBatches from '@/features/projects/project-batches'

export const Route = createLazyFileRoute(
  '/_authenticated/$orgSlug/$projectName/batches',
)({
  component: ProjectBatches,
})
