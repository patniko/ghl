import { createLazyFileRoute } from '@tanstack/react-router'
import ProjectConfig from '@/features/projects/project-config'

export const Route = createLazyFileRoute(
  '/_authenticated/$orgSlug/$projectName/config',
)({
  component: ProjectConfig,
})
