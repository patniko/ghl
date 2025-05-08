import { createLazyFileRoute } from '@tanstack/react-router'
import ProjectTasks from '@/features/projects/project-tasks'

export const Route = createLazyFileRoute(
  '/_authenticated/$orgSlug/$projectName/tasks',
)({
  component: ProjectTasks,
})
