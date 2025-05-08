import { useState, useEffect, useCallback } from 'react'
import ProjectAllFiles from './project-all-files'
import { useToast } from '@/hooks/use-toast'
import { Button } from '@/components/ui/button'
import { 
  ArrowLeftIcon, 
  PlusIcon, 
  RefreshCwIcon, 
  UploadIcon, 
  FileIcon, 
  DownloadIcon, 
  Trash2Icon,
  LayersIcon,
  FilesIcon
} from 'lucide-react'
import { Project } from '@/types/project'
import { Batch, BatchFile } from '@/types/batch'
import { projectService } from '@/services/projectService'
import { batchService, fileService } from '@/services/batchService'
import { Skeleton } from '@/components/ui/skeleton'
import { useNavigate, useParams } from '@tanstack/react-router'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Badge } from '@/components/ui/badge'
import { format } from 'date-fns'
import { UploadFileModal } from '@/components/ui/upload-file-modal'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"

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

export default function ProjectBatches() {
  const { toast } = useToast()
  const navigate = useNavigate()
  // Get organization slug and project name from route params
  const { orgSlug, projectName } = useParams({ 
    from: '/_authenticated/$orgSlug/$projectName/batches' 
  })

  // State for project, batches, and files
  const [project, setProject] = useState<Project | null>(null)
  const [batches, setBatches] = useState<Batch[]>([])
  const [currentBatch, setCurrentBatch] = useState<Batch | null>(null)
  const [batchFiles, setBatchFiles] = useState<BatchFile[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [isUploading, setIsUploading] = useState(false)
  const [isCreatingBatch, setIsCreatingBatch] = useState(false)
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)
  const [viewMode, setViewMode] = useState<'batches' | 'all-files'>('batches')
  const [isDeletingBatch, setIsDeletingBatch] = useState(false)
  const [batchToDelete, setBatchToDelete] = useState<Batch | null>(null)
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false)

  // Fetch project and batches
  const fetchProjectAndBatches = useCallback(async () => {
    setIsLoading(true)
    try {
      // Fetch project
      const projectData = await projectService.getProjectByName(projectName, orgSlug)
      setProject(projectData)

      // Fetch batches for this project
      const batchesData = await batchService.getBatches(orgSlug)
      const projectBatches = batchesData.filter(batch => batch.project_id === projectData.id)
      setBatches(projectBatches)

      // Set current batch to the first batch
      if (projectBatches.length > 0) {
        setCurrentBatch(projectBatches[0])
      }
    } catch (error) {
      console.error('Error fetching project and batches:', error)
      toast({
        title: 'Error',
        description: 'Failed to fetch project details',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }, [orgSlug, projectName, toast])

  // Fetch files for a batch
  const fetchBatchFiles = useCallback(async (batchId: number) => {
    try {
      const files = await batchService.getBatchFiles(batchId, undefined, orgSlug)
      setBatchFiles(files)
    } catch (error) {
      console.error('Error fetching batch files:', error)
      toast({
        title: 'Error',
        description: 'Failed to fetch batch files',
        variant: 'destructive',
      })
    }
  }, [orgSlug, toast])

  // Fetch project and batches on component mount
  useEffect(() => {
    fetchProjectAndBatches()
  }, [fetchProjectAndBatches])

  // Fetch batch files when current batch changes
  useEffect(() => {
    if (currentBatch) {
      fetchBatchFiles(currentBatch.id)
    }
  }, [currentBatch, fetchBatchFiles])


  // Handle batch change
  const handleBatchChange = (batchId: string) => {
    const selectedBatch = batches.find(b => b.id.toString() === batchId)
    if (selectedBatch) {
      setCurrentBatch(selectedBatch)
    }
  }

  // Handle file upload
  const handleFileUpload = async (files: File[]) => {
    if (!currentBatch) return

    setIsUploading(true)
    try {
      for (const file of files) {
        await fileService.uploadFile(file, currentBatch.id, undefined, orgSlug)
      }
      
      // Refresh files after upload
      await fetchBatchFiles(currentBatch.id)
      
      toast({
        title: 'Success',
        description: `${files.length} file(s) uploaded successfully`,
      })
    } catch (error) {
      console.error('Error uploading files:', error)
      toast({
        title: 'Error',
        description: 'Failed to upload files',
        variant: 'destructive',
      })
    } finally {
      setIsUploading(false)
    }
  }

  // Delete a file
  const handleDeleteFile = async (fileId: number) => {
    if (!currentBatch) return

    try {
      await fileService.deleteFile(fileId, orgSlug)
      
      // Refresh files after deletion
      await fetchBatchFiles(currentBatch.id)
      
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

  // Create a new batch
  const handleCreateNewBatch = async () => {
    if (!project) return

    setIsCreatingBatch(true)
    try {
      const newBatch = await batchService.createBatch(
        {
          name: `${project.name}-batch-${batches.length + 1}`,
          description: `Batch ${batches.length + 1} for project ${project.name}`,
          project_id: project.id
        },
        orgSlug
      )
      
      // Update batches list and set current batch to the new one
      setBatches(prev => [newBatch, ...prev])
      setCurrentBatch(newBatch)
      setBatchFiles([])
      
      toast({
        title: 'Success',
        description: 'New batch created successfully',
      })
    } catch (error) {
      console.error('Error creating new batch:', error)
      toast({
        title: 'Error',
        description: 'Failed to create new batch',
        variant: 'destructive',
      })
    } finally {
      setIsCreatingBatch(false)
    }
  }

  // Delete a batch
  const handleDeleteBatch = async () => {
    if (!batchToDelete) return

    setIsDeletingBatch(true)
    try {
      await batchService.deleteBatch(batchToDelete.id, orgSlug)
      
      // Update batches list
      setBatches(prev => prev.filter(b => b.id !== batchToDelete.id))
      
      // If the deleted batch was the current batch, set current batch to the first batch
      if (currentBatch && currentBatch.id === batchToDelete.id) {
        const remainingBatches = batches.filter(b => b.id !== batchToDelete.id)
        setCurrentBatch(remainingBatches.length > 0 ? remainingBatches[0] : null)
      }
      
      toast({
        title: 'Success',
        description: 'Batch deleted successfully',
      })
    } catch (error) {
      console.error('Error deleting batch:', error)
      toast({
        title: 'Error',
        description: 'Failed to delete batch',
        variant: 'destructive',
      })
    } finally {
      setIsDeletingBatch(false)
      setIsDeleteDialogOpen(false)
      setBatchToDelete(null)
    }
  }

  // Open delete batch dialog
  const openDeleteBatchDialog = (batch: Batch) => {
    setBatchToDelete(batch)
    setIsDeleteDialogOpen(true)
  }

  // Get file download URL
  const getFileDownloadUrl = (fileId: number): string => {
    return `${import.meta.env.VITE_API_URL || ''}/${orgSlug}/files/${fileId}/download`
  }

  return (
    <>
      {/* Back button - only visible on mobile */}
      <div className="mb-6 flex items-center md:hidden">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate({ 
            to: '/$orgSlug/$projectName', 
            params: { orgSlug, projectName } 
          })}
          className="mr-4"
        >
          <ArrowLeftIcon className="mr-2 h-4 w-4" />
          Back to Project
        </Button>
      </div>

      {isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-8 w-1/3" />
          <Skeleton className="h-4 w-1/2" />
          <Skeleton className="mt-6 h-[300px] w-full" />
        </div>
      ) : project ? (
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">{project.name} - Batches</h1>
            <p className="text-muted-foreground">
              Manage data batches for this project
            </p>
          </div>

          <Separator />

          {/* View switcher */}
          <div className="flex items-center space-x-2 mb-4">
            <Button
              variant={viewMode === 'batches' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('batches')}
              className="flex items-center"
            >
              <LayersIcon className="mr-2 h-4 w-4" />
              Batches
            </Button>
            <Button
              variant={viewMode === 'all-files' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setViewMode('all-files')}
              className="flex items-center"
            >
              <FilesIcon className="mr-2 h-4 w-4" />
              All Files
            </Button>
          </div>

          {viewMode === 'all-files' ? (
            <ProjectAllFiles orgSlug={orgSlug} projectId={project.id} />
          ) : (
            <>
              {/* Batch selector and controls */}
              <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex flex-1 items-center gap-2">
                  <Select
                    value={currentBatch?.id.toString()}
                    onValueChange={handleBatchChange}
                    disabled={batches.length === 0}
                  >
                    <SelectTrigger className="w-[250px]">
                      <SelectValue placeholder="Select a batch" />
                    </SelectTrigger>
                    <SelectContent>
                      {batches.map((batch) => (
                        <SelectItem key={batch.id} value={batch.id.toString()}>
                          {batch.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => currentBatch && fetchBatchFiles(currentBatch.id)}
                    disabled={!currentBatch}
                    title="Refresh files"
                  >
                    <RefreshCwIcon className="h-4 w-4" />
                  </Button>
                </div>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setIsUploadModalOpen(true)}
                    disabled={!currentBatch}
                  >
                    <UploadIcon className="mr-2 h-4 w-4" />
                    Upload Files
                  </Button>
                  <Button
                    onClick={handleCreateNewBatch}
                    disabled={isCreatingBatch || !project}
                  >
                    <PlusIcon className="mr-2 h-4 w-4" />
                    New Batch
                  </Button>
                </div>
              </div>

              {/* Current batch content */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <CardTitle>
                    {currentBatch ? `Batch: ${currentBatch.name}` : 'No Batch Selected'}
                  </CardTitle>
                  {currentBatch && (
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setIsUploadModalOpen(true)}
                      >
                        <UploadIcon className="mr-2 h-4 w-4" />
                        Upload
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openDeleteBatchDialog(currentBatch)}
                        className="text-destructive hover:text-destructive"
                      >
                        <Trash2Icon className="mr-2 h-4 w-4" />
                        Delete Batch
                      </Button>
                    </div>
                  )}
                </CardHeader>
                <CardContent>
                  {currentBatch ? (
                    <div className="space-y-6">
                      {/* Files list */}
                      {batchFiles.length > 0 ? (
                        <div className="space-y-3">
                          {batchFiles.map((file) => (
                            <Card key={file.id} className="overflow-hidden">
                              <div className="flex items-center justify-between p-4">
                                <div className="flex items-center space-x-4">
                                  <div className="rounded-md bg-muted p-2">
                                    <FileIcon className="h-6 w-6" />
                                  </div>
                                  <div>
                                    <div className="font-medium">{file.original_filename}</div>
                                    <div className="flex items-center space-x-2 text-sm text-muted-foreground">
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
                      ) : (
                        <div className="flex h-[200px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
                          <UploadIcon className="mb-4 h-10 w-10 text-muted-foreground" />
                          <p className="text-muted-foreground">
                            No files found in this batch.
                          </p>
                          <Button
                            variant="outline"
                            className="mt-4"
                            onClick={() => setIsUploadModalOpen(true)}
                          >
                            Upload Files
                          </Button>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="flex h-[300px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
                      <p className="text-muted-foreground">
                        {batches.length > 0
                          ? 'Select a batch to view and manage files'
                          : 'No batches found for this project. Create a new batch to get started.'}
                      </p>
                      {batches.length === 0 && (
                        <Button
                          className="mt-4"
                          onClick={handleCreateNewBatch}
                          disabled={isCreatingBatch || !project}
                        >
                          <PlusIcon className="mr-2 h-4 w-4" />
                          Create Batch
                        </Button>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          )}
        </div>
      ) : (
        <div className="flex h-[300px] flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center">
          <h3 className="mb-2 text-lg font-medium">Project not found</h3>
          <p className="mb-4 text-sm text-muted-foreground">
            The project you're looking for doesn't exist or you don't have access to it.
          </p>
          <Button
            onClick={() => navigate({ to: '/$orgSlug', params: { orgSlug } })}
            className="flex items-center"
          >
            <ArrowLeftIcon className="mr-2 h-4 w-4" />
            Back to Projects
          </Button>
        </div>
      )}

      {/* Upload file modal */}
      <UploadFileModal
        open={isUploadModalOpen}
        onOpenChange={setIsUploadModalOpen}
        onUpload={handleFileUpload}
        isUploading={isUploading}
        title="Upload Files"
        description="Select files to upload to this batch"
      />

      {/* Delete batch confirmation dialog */}
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete the batch "{batchToDelete?.name}" and all its files.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteBatch}
              disabled={isDeletingBatch}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeletingBatch ? 'Deleting...' : 'Delete Batch'}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
