import { createLazyFileRoute } from '@tanstack/react-router'
import SyntheticSets from '@/features/synthetic-sets'

export const Route = createLazyFileRoute('/_authenticated/$orgSlug/synthetic-sets')({
  component: SyntheticSets,
})

// Also export the component directly for testing
export default SyntheticSets
