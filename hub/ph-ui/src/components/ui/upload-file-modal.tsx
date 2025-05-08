import { useState } from 'react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from '@/components/ui/dialog'
import { DropZone } from '@/components/ui/drop-zone'

interface UploadFileModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onUpload: (files: File[]) => Promise<void>
  isUploading: boolean
  title?: string
  description?: string
  acceptedFileTypes?: Record<string, string[]>
}

export function UploadFileModal({
  open,
  onOpenChange,
  onUpload,
  isUploading,
  title = 'Upload Files',
  description = 'Drag and drop files or click to browse',
  acceptedFileTypes,
}: UploadFileModalProps) {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])

  const handleDrop = (files: File[]) => {
    setSelectedFiles(files)
  }

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return
    
    try {
      await onUpload(selectedFiles)
      setSelectedFiles([])
      onOpenChange(false)
    } catch (error) {
      console.error('Error uploading files:', error)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          <DialogDescription>{description}</DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          <DropZone
            onDrop={handleDrop}
            accept={acceptedFileTypes}
            disabled={isUploading}
            className="h-32"
            maxSize={5 * 1024 * 1024 * 1024} // 5GB max size
          />
          
          {selectedFiles.length > 0 && (
            <div className="mt-4 space-y-2 rounded-md border p-3">
              <h4 className="text-sm font-medium">Selected Files:</h4>
              <ul className="max-h-32 space-y-1 overflow-auto text-sm">
                {selectedFiles.map((file, index) => (
                  <li key={index} className="flex justify-between">
                    <span className="truncate">{file.name}</span>
                    <span className="text-muted-foreground">{formatFileSize(file.size)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        
        <DialogFooter className="flex items-center justify-between">
          <DialogClose asChild>
            <Button variant="outline" disabled={isUploading}>
              Cancel
            </Button>
          </DialogClose>
          <Button 
            onClick={handleUpload} 
            disabled={selectedFiles.length === 0 || isUploading}
            className="ml-2"
          >
            {isUploading ? 'Uploading...' : 'Upload'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
