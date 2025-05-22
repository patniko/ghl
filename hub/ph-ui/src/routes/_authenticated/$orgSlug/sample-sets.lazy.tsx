import { createLazyFileRoute } from '@tanstack/react-router'
import SampleSets from '@/features/sample-sets'

export const Route = createLazyFileRoute(
  '/_authenticated/$orgSlug/sample-sets',
)({
  component: SampleSets,
})

// Also export the component directly for testing
export default SampleSets
