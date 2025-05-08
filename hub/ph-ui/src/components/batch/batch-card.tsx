import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Batch } from '@/types/batch'
import { FileIcon, UploadIcon, Trash2Icon } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

interface BatchCardProps {
  batch: Batch
  onViewBatch: (batch: Batch) => void
  onUploadFile: (batch: Batch) => void
  onDeleteBatch: (batch: Batch) => void
  filesCount: number
}

export function BatchCard({
  batch,
  onViewBatch,
  onUploadFile,
  onDeleteBatch,
  filesCount,
}: BatchCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div>
            <CardTitle className="text-xl">{batch.name}</CardTitle>
            <CardDescription className="mt-1 line-clamp-2">
              {batch.description || 'No description provided'}
            </CardDescription>
          </div>
          <Badge variant="outline" className="ml-2 mt-1">
            {filesCount} {filesCount === 1 ? 'file' : 'files'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pb-2">
        <div className="flex flex-col space-y-1 text-sm text-muted-foreground">
          <div className="flex items-center">
            <FileIcon className="mr-2 h-4 w-4" />
            <span>
              Created {formatDistanceToNow(new Date(batch.created_at), { addSuffix: true })}
            </span>
          </div>
          {batch.quality_summary && (
            <div className="flex items-center mt-2">
              <div className="flex items-center space-x-4">
                <div className="flex items-center">
                  <Badge variant="outline" className="mr-2">
                    {batch.quality_summary.total_checks} checks
                  </Badge>
                </div>
                {batch.quality_summary.issues_by_severity.error > 0 && (
                  <Badge variant="destructive">
                    {batch.quality_summary.issues_by_severity.error} errors
                  </Badge>
                )}
                {batch.quality_summary.issues_by_severity.warning > 0 && (
                  <Badge variant="secondary">
                    {batch.quality_summary.issues_by_severity.warning} warnings
                  </Badge>
                )}
              </div>
            </div>
          )}
        </div>
      </CardContent>
      <CardFooter className="pt-3">
        <div className="flex w-full justify-between">
          <Button variant="outline" size="sm" onClick={() => onViewBatch(batch)}>
            View Details
          </Button>
          <div className="flex space-x-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => onUploadFile(batch)}
              className="flex items-center"
            >
              <UploadIcon className="mr-1 h-4 w-4" />
              Upload
            </Button>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => onDeleteBatch(batch)}
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
