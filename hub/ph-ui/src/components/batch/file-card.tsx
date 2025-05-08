import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { BatchFile } from '@/types/batch'
import { DownloadIcon, EyeIcon, Trash2Icon } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'
import { fileService } from '@/services/batchService'

interface FileCardProps {
  file: BatchFile
  onViewFile: (file: BatchFile) => void
  onDeleteFile: (file: BatchFile) => void
}

export function FileCard({ file, onViewFile, onDeleteFile }: FileCardProps) {
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB'
  }

  const getFileTypeColor = (fileType: string): string => {
    switch (fileType.toLowerCase()) {
      case 'csv':
        return 'bg-green-100 text-green-800'
      case 'mp4':
        return 'bg-blue-100 text-blue-800'
      case 'npz':
        return 'bg-purple-100 text-purple-800'
      case 'dicom':
        return 'bg-orange-100 text-orange-800'
      case 'json':
        return 'bg-yellow-100 text-yellow-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

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

  const isPreviewable = ['csv', 'dicom', 'npz'].includes(file.file_type.toLowerCase())

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between">
          <CardTitle className="text-base truncate" title={file.original_filename}>
            {file.original_filename}
          </CardTitle>
          <div className="flex space-x-2">
            <Badge className={getFileTypeColor(file.file_type)}>
              {file.file_type.toUpperCase()}
            </Badge>
            <Badge className={getStatusColor(file.processing_status)}>
              {file.processing_status.toUpperCase()}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pb-2">
        <div className="flex flex-col space-y-1 text-xs text-muted-foreground">
          <div className="flex justify-between">
            <span>Size: {formatFileSize(file.file_size)}</span>
            <span>
              Uploaded {formatDistanceToNow(new Date(file.created_at), { addSuffix: true })}
            </span>
          </div>
          {file.csv_headers && (
            <div className="mt-2">
              <span className="font-medium">Headers:</span>{' '}
              <span className="truncate">
                {file.csv_headers.slice(0, 5).join(', ')}
                {file.csv_headers.length > 5 ? '...' : ''}
              </span>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="pt-2">
        <div className="flex w-full justify-between">
          {isPreviewable ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => onViewFile(file)}
              className="flex items-center"
            >
              <EyeIcon className="mr-1 h-4 w-4" />
              View
            </Button>
          ) : (
            <div></div>
          )}
          <div className="flex space-x-2">
            <Button
              variant="outline"
              size="sm"
              asChild
              className="flex items-center"
            >
              <a
                href={fileService.getFileDownloadUrl(file.id)}
                target="_blank"
                rel="noopener noreferrer"
              >
                <DownloadIcon className="mr-1 h-4 w-4" />
                Download
              </a>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onDeleteFile(file)}
              className="flex items-center text-destructive hover:text-destructive"
            >
              <Trash2Icon className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardFooter>
    </Card>
  )
}
