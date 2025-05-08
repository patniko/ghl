import { useState, useEffect, useCallback } from 'react'
import { useToast } from '@/hooks/use-toast'
import { Button } from '@/components/ui/button'
import { 
  FileIcon, 
  DownloadIcon, 
  Trash2Icon,
  SearchIcon,
  ArrowUpDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon
} from 'lucide-react'
import { batchService, fileService, AllFilesParams, AllFilesResponse } from '@/services/batchService'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuLabel
} from '@/components/ui/dropdown-menu'
import { Badge } from '@/components/ui/badge'
import { format } from 'date-fns'
import { Skeleton } from '@/components/ui/skeleton'

// Utility functions for formatting
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

const formatFileDate = (dateString: string): string => {
  try {
    return format(new Date(dateString), 'MMM d, yyyy h:mm a')
  } catch (_) {
    return dateString
  }
}

// Get file type color
const getFileTypeColor = (fileType: string): string => {
  switch (fileType.toLowerCase()) {
    case 'csv':
      return 'bg-green-100 text-green-800'
    case 'dicom':
      return 'bg-orange-100 text-orange-800'
    case 'mp4':
      return 'bg-blue-100 text-blue-800'
    case 'npz':
      return 'bg-purple-100 text-purple-800'
    case 'json':
      return 'bg-yellow-100 text-yellow-800'
    default:
      return 'bg-gray-100 text-gray-800'
  }
}

// Get schema type color
const getSchemaTypeColor = (schemaType: string): string => {
  switch (schemaType.toLowerCase()) {
    case 'alivecor':
      return 'bg-pink-100 text-pink-800'
    default:
      return 'bg-indigo-100 text-indigo-800'
  }
}

// Get processing status color
const getStatusColor = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'completed':
      return 'bg-green-100 text-green-800'
    case 'processing':
      return 'bg-blue-100 text-blue-800'
    case 'failed':
      return 'bg-red-100 text-red-800'
    default:
      return 'bg-yellow-100 text-yellow-800'
  }
}

interface AllFilesViewProps {
  orgSlug: string
  projectId?: number
}

export default function AllFilesView({ orgSlug, projectId }: AllFilesViewProps) {
  const { toast } = useToast()
  
  // State for files and pagination
  const [filesResponse, setFilesResponse] = useState<AllFilesResponse | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState<AllFilesParams['sort_by']>('created_at')
  const [sortOrder, setSortOrder] = useState<AllFilesParams['sort_order']>('desc')
  const [currentPage, setCurrentPage] = useState(1)
  const [fileTypeFilter, setFileTypeFilter] = useState<string | undefined>(undefined)
  
  // Fetch files
  const fetchFiles = useCallback(async () => {
    setIsLoading(true)
    try {
      const params: AllFilesParams = {
        page: currentPage,
        page_size: 10,
        sort_by: sortBy,
        sort_order: sortOrder,
        search: debouncedSearchTerm || undefined,
        file_type: fileTypeFilter,
      }
      
      if (projectId) {
        params.project_id = projectId
      }
      
      const response = await batchService.getAllFiles(params, orgSlug)
      setFilesResponse(response)
    } catch (error) {
      console.error('Error fetching files:', error)
      toast({
        title: 'Error',
        description: 'Failed to fetch files',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }, [orgSlug, projectId, debouncedSearchTerm, sortBy, sortOrder, currentPage, fileTypeFilter, toast])
  
  // Debounce search term
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm)
    }, 500)
    
    return () => clearTimeout(timer)
  }, [searchTerm])
  
  // Fetch files when params change
  useEffect(() => {
    fetchFiles()
  }, [fetchFiles])
  
  // Delete a file
  const handleDeleteFile = async (fileId: number) => {
    try {
      await fileService.deleteFile(fileId, orgSlug)
      
      // Refresh files after deletion
      await fetchFiles()
      
      toast({
        title: 'Success',
        description: 'File deleted successfully',
      })
    } catch (error) {
      console.error('Error deleting file:', error)
      toast({
        title: 'Error',
        description: 'Failed to delete file',
        variant: 'destructive',
      })
    }
  }
  
  // Get file download URL
  const getFileDownloadUrl = (fileId: number): string => {
    return `${import.meta.env.VITE_API_URL || ''}/${orgSlug}/files/${fileId}/download`
  }
  
  // Handle pagination
  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1)
    }
  }
  
  const handleNextPage = () => {
    if (filesResponse && currentPage < filesResponse.total_pages) {
      setCurrentPage(currentPage + 1)
    }
  }
  
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle>All Files</CardTitle>
        <div className="flex items-center space-x-2">
          {/* Search input */}
          <div className="relative">
            <SearchIcon className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search files..."
              className="w-[200px] pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          
          {/* Sort dropdown */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm">
                <ArrowUpDownIcon className="mr-2 h-4 w-4" />
                Sort
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-[200px]">
              <DropdownMenuLabel>Sort by</DropdownMenuLabel>
              <DropdownMenuRadioGroup value={sortBy} onValueChange={(value) => setSortBy(value as AllFilesParams['sort_by'])}>
                <DropdownMenuRadioItem value="created_at">Date</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="filename">Filename</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="file_size">Size</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="file_type">Type</DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Order</DropdownMenuLabel>
              <DropdownMenuRadioGroup value={sortOrder} onValueChange={(value) => setSortOrder(value as AllFilesParams['sort_order'])}>
                <DropdownMenuRadioItem value="desc">Descending</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="asc">Ascending</DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
              <DropdownMenuSeparator />
              <DropdownMenuLabel>Filter by type</DropdownMenuLabel>
              <DropdownMenuRadioGroup 
                value={fileTypeFilter || 'all'} 
                onValueChange={(value) => setFileTypeFilter(value === 'all' ? undefined : value)}
              >
                <DropdownMenuRadioItem value="all">All types</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="csv">CSV</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="dicom">DICOM</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="mp4">MP4</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="npz">NPZ</DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="json">JSON</DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-20 w-full" />
            ))}
          </div>
        ) : filesResponse && filesResponse.files.length > 0 ? (
          <>
            <div className="space-y-3">
              {filesResponse.files.map((file) => (
                <Card key={file.id} className="overflow-hidden">
                  <div className="flex items-center justify-between p-4">
                    <div className="flex items-center space-x-4">
                      <div className="rounded-md bg-muted p-2">
                        <FileIcon className="h-6 w-6" />
                      </div>
                      <div>
                        <div className="font-medium">{file.original_filename}</div>
                        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                          <Badge className={getFileTypeColor(file.file_type)}>
                            {file.file_type.toUpperCase()}
                          </Badge>
                          <Badge className={getStatusColor(file.processing_status)}>
                            {file.processing_status.toUpperCase()}
                          </Badge>
                          {file.schema_type && (
                            <Badge className={getSchemaTypeColor(file.schema_type)}>
                              {file.schema_type.toUpperCase()}
                            </Badge>
                          )}
                          <span>{formatFileSize(file.file_size)}</span>
                          <span>•</span>
                          <span>{formatFileDate(file.created_at)}</span>
                          {file.batch_name && (
                            <>
                              <span>•</span>
                              <span>Batch: {file.batch_name}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="h-4 w-4"
                          >
                            <circle cx="12" cy="12" r="1" />
                            <circle cx="12" cy="5" r="1" />
                            <circle cx="12" cy="19" r="1" />
                          </svg>
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem asChild>
                          <a 
                            href={getFileDownloadUrl(file.id)} 
                            target="_blank" 
                            rel="noopener noreferrer"
                            className="flex cursor-pointer items-center"
                          >
                            <DownloadIcon className="mr-2 h-4 w-4" />
                            Download
                          </a>
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem 
                          className="flex cursor-pointer items-center text-destructive focus:text-destructive"
                          onClick={() => handleDeleteFile(file.id)}
                        >
                          <Trash2Icon className="mr-2 h-4 w-4" />
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </Card>
              ))}
            </div>
            
            {/* Pagination */}
            <div className="mt-4 flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {filesResponse.files.length} of {filesResponse.total} files
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePreviousPage}
                  disabled={currentPage === 1}
                >
                  <ChevronLeftIcon className="h-4 w-4" />
                </Button>
                <div className="text-sm">
                  Page {currentPage} of {filesResponse.total_pages}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleNextPage}
                  disabled={currentPage === filesResponse.total_pages}
                >
                  <ChevronRightIcon className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex h-[200px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
            <p className="text-muted-foreground">
              No files found.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
