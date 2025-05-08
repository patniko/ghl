import { useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Input } from '@/components/ui/input'
import { Batch } from '@/types/batch'

interface UploadFileModalProps {
  isOpen: boolean
  onClose: () => void
  onUploadFile: (file: File, batchId: number) => Promise<void>
  isLoading: boolean
  batch: Batch | null
}

export function UploadFileModal({
  isOpen,
  onClose,
  onUploadFile,
  isLoading,
  batch,
}: UploadFileModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (selectedFile && batch) {
      await onUploadFile(selectedFile, batch.id)
      setSelectedFile(null)
    }
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <form onSubmit={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Upload File</DialogTitle>
            <DialogDescription>
              Upload a file to the batch {batch?.name}.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="file">Select File</Label>
              <Input
                id="file"
                type="file"
                onChange={handleFileChange}
                accept=".csv,.mp4,.npz,.dcm,.dicom,.json"
                className="cursor-pointer"
              />
              <p className="text-sm text-muted-foreground">
                Supported file types: CSV, MP4, NPZ, DICOM, JSON
              </p>
            </div>

            {selectedFile && (
              <div className="rounded-md bg-muted p-3">
                <div className="text-sm font-medium">Selected File:</div>
                <div className="text-sm">{selectedFile.name}</div>
                <div className="text-xs text-muted-foreground">
                  Size: {formatFileSize(selectedFile.size)}
                </div>
              </div>
            )}
          </div>
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isLoading}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={!selectedFile || isLoading}>
              {isLoading ? 'Uploading...' : 'Upload'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
