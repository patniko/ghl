import { useState, useEffect, useCallback } from 'react'
import { useToast } from '@/hooks/use-toast'
import { Header } from '@/components/layout/header'
import { Main } from '@/components/layout/main'
import { ProfileDropdown } from '@/components/profile-dropdown'
import { Search } from '@/components/search'
import { ThemeSwitch } from '@/components/theme-switch'
import { Button } from '@/components/ui/button'
import { PlusIcon, DatabaseIcon } from 'lucide-react'
import { SyntheticDataset, SyntheticDatasetCreate, SamplesDatasetCreate } from '@/types/synthetic-dataset'
import { syntheticDatasetsService } from '@/services/syntheticDatasetsService'
import { SyntheticSetCard } from '@/components/synthetic-set/synthetic-set-card'
import { CreateSyntheticSetModal } from '@/components/synthetic-set/create-synthetic-set-modal'
import { DeleteSyntheticSetDialog } from '@/components/synthetic-set/delete-synthetic-set-dialog'
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

export default function SyntheticSets() {
  const { toast } = useToast()
  const _navigate = useNavigate()
  // Get organization slug from route params
  const { orgSlug } = useParams({ from: '/_authenticated/$orgSlug/synthetic-sets' })

  // State for datasets
  const [datasets, setDatasets] = useState<SyntheticDataset[]>([])
  const [isLoadingDatasets, setIsLoadingDatasets] = useState(true)
  const [selectedDataset, setSelectedDataset] = useState<SyntheticDataset | null>(null)

  // State for modals
  const [isCreateDatasetModalOpen, setIsCreateDatasetModalOpen] = useState(false)
  const [isDeleteDatasetModalOpen, setIsDeleteDatasetModalOpen] = useState(false)

  // State for loading states
  const [isCreatingDataset, setIsCreatingDataset] = useState(false)
  const [isDeletingDataset, setIsDeletingDataset] = useState(false)
  const [isLoadingSamplesDatasets, setIsLoadingSamplesDatasets] = useState(true)
  const [samplesDatasets, setSamplesDatasets] = useState<SyntheticDataset[]>([])

  // State for filtering and sorting
  const [sort, setSort] = useState('descending')
  const [searchTerm, setSearchTerm] = useState('')

  // Fetch datasets
  const fetchDatasets = useCallback(async () => {
    setIsLoadingDatasets(true)
    try {
      const data = await syntheticDatasetsService.getDatasets(orgSlug)
      setDatasets(data)
    } catch (error) {
      console.error('Error fetching datasets:', error)
      toast({
        title: 'Error',
        description: 'Failed to fetch synthetic datasets',
        variant: 'destructive',
      })
    } finally {
      setIsLoadingDatasets(false)
    }
  }, [orgSlug, toast])
  
  // Fetch samples datasets
  const fetchSamplesDatasets = useCallback(async () => {
    setIsLoadingSamplesDatasets(true)
    try {
      const data = await syntheticDatasetsService.getSamplesDatasets(orgSlug)
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
    fetchDatasets()
    fetchSamplesDatasets()
  }, [fetchDatasets, fetchSamplesDatasets])

  // Create dataset
  const handleCreateDataset = async (dataset: SyntheticDatasetCreate, file?: File) => {
    setIsCreatingDataset(true)
    try {
      const newDataset = await syntheticDatasetsService.createDataset(dataset, orgSlug)
      
      // If a file was provided, upload it
      if (file) {
        await syntheticDatasetsService.uploadCsv(newDataset.id, file, orgSlug)
      }
      
      setDatasets((prev) => [newDataset, ...prev])
      setIsCreateDatasetModalOpen(false)
      toast({
        title: 'Success',
        description: 'Dataset created successfully',
      })
    } catch (error) {
      console.error('Error creating dataset:', error)
      toast({
        title: 'Error',
        description: 'Failed to create dataset',
        variant: 'destructive',
      })
    } finally {
      setIsCreatingDataset(false)
    }
  }
  
  // Create samples dataset
  const handleCreateSamplesDataset = async (dataset: SamplesDatasetCreate) => {
    setIsCreatingDataset(true)
    try {
      const newDataset = await syntheticDatasetsService.createSamplesDataset(dataset, orgSlug)
      
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
      // Check if it's a samples dataset
      const isSamplesDataset = samplesDatasets.some(d => d.id === selectedDataset.id)
      
      if (isSamplesDataset) {
        await syntheticDatasetsService.deleteSamplesDataset(selectedDataset.id, orgSlug)
        setSamplesDatasets((prev) => prev.filter((dataset) => dataset.id !== selectedDataset.id))
      } else {
        await syntheticDatasetsService.deleteDataset(selectedDataset.id, orgSlug)
        setDatasets((prev) => prev.filter((dataset) => dataset.id !== selectedDataset.id))
      }
      
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

  // Download dataset as CSV
  const handleDownloadDataset = async (dataset: SyntheticDataset) => {
    try {
      // Check if it's a samples dataset
      const isSamplesDataset = samplesDatasets.some(d => d.id === dataset.id)
      
      if (isSamplesDataset) {
        // Download samples dataset as zip
        const blob = await syntheticDatasetsService.downloadSamplesDataset(dataset.id, orgSlug)
        
        // Create download link
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${dataset.name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.zip`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      } else {
        // Get all data (this is a simplified approach - in a real app you might want to handle pagination)
        const response = await syntheticDatasetsService.getDatasetData(dataset.id, orgSlug, 1, 10000)
        
        if (!response.data || response.data.length === 0) {
          toast({
            title: 'No data',
            description: 'This dataset contains no data to download',
            variant: 'destructive',
          })
          return
        }
        
        // Convert to CSV
        const headers = Object.keys(response.data[0]).join(',')
        const rows = response.data.map(row => 
          Object.values(row).map(value => 
            typeof value === 'string' ? `"${value.replace(/"/g, '""')}"` : value
          ).join(',')
        )
        const csv = [headers, ...rows].join('\n')
        
        // Create download link
        const blob = new Blob([csv], { type: 'text/csv' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `${dataset.name.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.csv`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
      }
      
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
  const filteredDatasets = [...datasets, ...samplesDatasets]
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
            <h2 className="text-2xl font-bold tracking-tight">Synthetic Sets</h2>
            <p className="text-muted-foreground">
              Create and manage synthetic datasets for testing and development
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

        {isLoadingDatasets || isLoadingSamplesDatasets ? (
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
              Create your first synthetic dataset to start generating test data
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
              <SyntheticSetCard
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
      <CreateSyntheticSetModal
        isOpen={isCreateDatasetModalOpen}
        onClose={() => setIsCreateDatasetModalOpen(false)}
        onCreateDataset={handleCreateDataset}
        onCreateSamplesDataset={handleCreateSamplesDataset}
        isLoading={isCreatingDataset}
      />

      <DeleteSyntheticSetDialog
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
