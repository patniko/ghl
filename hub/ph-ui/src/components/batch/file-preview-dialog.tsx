import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { BatchFile } from '@/types/batch'
import { CsvPreview } from './csv-preview'
import { useState, useEffect } from 'react'
import { fileService } from '@/services/batchService'
import { Skeleton } from '@/components/ui/skeleton'

interface FilePreviewDialogProps {
  isOpen: boolean
  onClose: () => void
  file: BatchFile | null
}

export function FilePreviewDialog({
  isOpen,
  onClose,
  file,
}: FilePreviewDialogProps) {
  const [csvData, setCsvData] = useState<{
    headers: string[]
    data: string[][]
    total_rows: number
    preview_rows: number
  } | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (isOpen && file && file.file_type === 'csv') {
      setIsLoading(true)
      fileService
        .getCsvPreview(file.id)
        .then((data) => {
          setCsvData(data)
        })
        .catch((error) => {
          console.error('Error loading CSV preview:', error)
          setCsvData(null)
        })
        .finally(() => {
          setIsLoading(false)
        })
    } else {
      setCsvData(null)
    }
  }, [isOpen, file])

  if (!file) return null

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle>{file.original_filename}</DialogTitle>
          <DialogDescription>
            File type: {file.file_type.toUpperCase()} | Size:{' '}
            {formatFileSize(file.file_size)}
          </DialogDescription>
        </DialogHeader>

        {file.file_type === 'csv' && (
          <>
            {isLoading ? (
              <div className="space-y-2">
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-[400px] w-full" />
              </div>
            ) : (
              <CsvPreview data={csvData} />
            )}
          </>
        )}

        {file.file_type === 'dicom' && (
          <div className="flex justify-center">
            <img
              src={`${fileService.getFileDownloadUrl(file.id)}`}
              alt="DICOM Preview"
              className="max-h-[500px] object-contain"
            />
          </div>
        )}

        {file.file_type !== 'csv' && file.file_type !== 'dicom' && (
          <div className="flex h-[300px] items-center justify-center">
            <p className="text-muted-foreground">
              Preview not available for this file type
            </p>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}

// Helper function to format file size
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
}
