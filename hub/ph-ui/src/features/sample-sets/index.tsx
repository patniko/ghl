import { useState, useEffect, useCallback } from 'react'
import { useToast } from '@/hooks/use-toast'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { Button } from '@/components/ui/button'
import { PlusIcon, DatabaseIcon } from 'lucide-react'
import { SampleDataset, SamplesDatasetCreate } from '@/types/sample-dataset'
import { samplesService } from '@/services/samplesService'
import { SampleSetCard } from '@/components/sample-set/sample-set-card'
import { CreateSampleSetModal } from '@/components/sample-set/create-sample-set-modal'
import { DeleteSampleSetDialog } from '@/components/sample-set/delete-sample-set-dialog'
import { Skeleton } from '@/components/ui/skeleton'
import { useNavigate, useParams } from '@tanstack/react-router'
import {
  IconAdjustmentsHorizontal,
  IconSortAscendingLetters,
  IconSortDescendingLetters,
} from '@tabler/icons-react'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Separator } from '@/components/ui/separator'

export default function SampleSets() {
  const { toast } = useToast()
  const _navigate = useNavigate()
  // Get organization slug from route params
  const { orgSlug } = useParams({ from: '/_authenticated/$orgSlug/sample-sets' })

  // State for datasets
  const [selectedDataset, setSelectedDataset] = useState<SampleDataset | null>(null)

  // State for modals
  const [isCreateDatasetModalOpen, setIsCreateDatasetModalOpen] = useState(false)
  const [isDeleteDatasetModalOpen, setIsDeleteDatasetModalOpen] = useState(false)

  // State for loading states
  const [isCreatingDataset, setIsCreatingDataset] = useState(false)
  const [isDeletingDataset, setIsDeletingDataset] = useState(false)
  const [isLoadingSamplesDatasets, setIsLoadingSamplesDatasets] = useState(true)
  const [samplesDatasets, setSamplesDatasets] = useState<SampleDataset[]>([])

  // State for filtering and sorting
  const [sort, setSort] = useState('descending')
  const [searchTerm, setSearchTerm] = useState('')

  // Fetch samples datasets
  const fetchSamplesDatasets = useCallback(async () => {
    setIsLoadingSamplesDatasets(true)
    try {
      const data = await samplesService.getSamplesDatasets(orgSlug)
      setSamplesDatasets(data)
    } catch (error) {
      console.error('Error fetching samples datasets:', error)
      toast({
        title: 'Error',
        description: 'Failed to fetch samples datasets',
        variant: 'destructive',
      })
    } finally {
      setIsLoadingSamplesDatasets(false)
    }
  }, [orgSlug, toast])
  
  // Fetch datasets on component mount or when orgSlug changes
  useEffect(() => {
    fetchSamplesDatasets()
  }, [fetchSamplesDatasets])

  // Create samples dataset
  const handleCreateSamplesDataset = async (dataset: SamplesDatasetCreate) => {
    setIsCreatingDataset(true)
    try {
      const newDataset = await samplesService.createSamplesDataset(dataset, orgSlug)
      
      setSamplesDatasets((prev) => [newDataset, ...prev])
      setIsCreateDatasetModalOpen(false)
      toast({
        title: 'Success',
        description: 'Samples dataset created successfully',
      })
    } catch (error) {
      console.error('Error creating samples dataset:', error)
      toast({
        title: 'Error',
        description: 'Failed to create samples dataset',
        variant: 'destructive',
      })
    } finally {
      setIsCreatingDataset(false)
    }
  }

  // Delete dataset
  const handleDeleteDataset = async () => {
    if (!selectedDataset) return

    setIsDeletingDataset(true)
    try {
      await samplesService.deleteSamplesDataset(selectedDataset.id, orgSlug)
      setSamplesDatasets((prev) => prev.filter((dataset) => dataset.id !== selectedDataset.id))
      
      setIsDeleteDatasetModalOpen(false)
      toast({
        title: 'Success',
        description: 'Dataset deleted successfully',
      })
    } catch (error) {
      console.error('Error deleting dataset:', error)
      toast({
        title: 'Error',
        description: 'Failed to delete dataset',
        variant: 'destructive',
      })
    } finally {
      setIsDeletingDataset(false)
    }
  }

  // Download dataset as ZIP
  const handleDownloadDataset = async (dataset: SampleDataset) => {
    try {
      // Download samples dataset as zip
      const blob = await samplesService.downloadSamplesDataset(dataset.id, orgSlug)
      
      // Create download link
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${dataset.name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.zip`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      toast({
        title: 'Success',
        description: 'Dataset downloaded successfully',
      })
    } catch (error) {
      console.error('Error downloading dataset:', error)
      toast({
        title: 'Error',
        description: 'Failed to download dataset',
        variant: 'destructive',
      })
    }
  }

  // Filter and sort datasets
  const filteredDatasets = [...samplesDatasets]
    .filter((dataset) => {
      // Filter by search term
      return dataset.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (dataset.description && dataset.description.toLowerCase().includes(searchTerm.toLowerCase()))
    })
    .sort((a, b) => {
      // Sort by creation date
      const dateA = new Date(a.created_at).getTime()
      const dateB = new Date(b.created_at).getTime()
      return sort === 'ascending' ? dateA - dateB : dateB - dateA
    })

  return (
    <>
      <Header fixed>
        <Search />
        <div className="ml-auto flex items-center space-x-4">
          <ThemeSwitch />
          <ProfileDropdown />
        </div>
      </Header>

      <Main>
        <div className="mb-6 flex flex-wrap items-center justify-between gap-x-4 space-y-2">
          <div>
            <h2 className="text-2xl font-bold tracking-tight">Sample Sets</h2>
            <p className="text-muted-foreground">
              Create and manage sample datasets for testing and development
            </p>
          </div>
          <Button
            onClick={() => setIsCreateDatasetModalOpen(true)}
            className="flex items-center"
          >
            <PlusIcon className="mr-2 h-4 w-4" />
            Create Dataset
          </Button>
        </div>

        <div className="my-4 flex items-end justify-between sm:my-0 sm:items-center">
          <div className="flex flex-col gap-4 sm:my-4 sm:flex-row">
            <Input
              placeholder="Search datasets..."
              className="h-9 w-40 lg:w-[250px]"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <Select value={sort} onValueChange={setSort}>
            <SelectTrigger className="w-16">
              <SelectValue>
                <IconAdjustmentsHorizontal size={18} />
              </SelectValue>
            </SelectTrigger>
            <SelectContent align="end">
              <SelectItem value="ascending">
                <div className="flex items-center gap-4">
                  <IconSortAscendingLetters size={16} />
                  <span>Oldest First</span>
                </div>
              </SelectItem>
              <SelectItem value="descending">
                <div className="flex items-center gap-4">
                  <IconSortDescendingLetters size={16} />
                  <span>Newest First</span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator className="shadow" />

        {isLoadingSamplesDatasets ? (
          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-[200px] w-full" />
            ))}
          </div>
        ) : filteredDatasets.length === 0 ? (
          <div className="flex h-[300px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
            <DatabaseIcon className="mb-4 h-12 w-12 text-muted-foreground" />
            <h3 className="mb-2 text-lg font-medium">No datasets found</h3>
            <p className="mb-4 text-sm text-muted-foreground">
              Create your first sample dataset to start generating test data
            </p>
            <Button
              onClick={() => setIsCreateDatasetModalOpen(true)}
              className="flex items-center"
            >
              <PlusIcon className="mr-2 h-4 w-4" />
              Create Dataset
            </Button>
          </div>
        ) : (
          <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
            {filteredDatasets.map((dataset) => (
              <SampleSetCard
                key={dataset.id}
                dataset={dataset}
                onViewDataset={(dataset) => {
                  // In a real app, you would navigate to a details page
                  toast({
                    title: 'View Dataset',
                    description: `Viewing dataset: ${dataset.name}`,
                  })
                }}
                onEditDataset={(dataset) => {
                  // In a real app, you would open an edit modal
                  toast({
                    title: 'Edit Dataset',
                    description: `Editing dataset: ${dataset.name}`,
                  })
                }}
                onDeleteDataset={(dataset) => {
                  setSelectedDataset(dataset)
                  setIsDeleteDatasetModalOpen(true)
                }}
                onDownloadDataset={handleDownloadDataset}
              />
            ))}
          </div>
        )}
      </Main>

      {/* Modals */}
      <CreateSampleSetModal
        isOpen={isCreateDatasetModalOpen}
        onClose={() => setIsCreateDatasetModalOpen(false)}
        _onCreateDataset={() => Promise.resolve()} // Dummy function since we only use samples datasets
        onCreateSamplesDataset={handleCreateSamplesDataset}
        isLoading={isCreatingDataset}
      />

      <DeleteSampleSetDialog
        isOpen={isDeleteDatasetModalOpen}
        onClose={() => setIsDeleteDatasetModalOpen(false)}
        onConfirm={handleDeleteDataset}
        title="Delete Dataset"
        description={`Are you sure you want to delete the dataset "${selectedDataset?.name}"? This action cannot be undone and all associated data will be deleted.`}
        isLoading={isDeletingDataset}
      />
    </>
  )
}
