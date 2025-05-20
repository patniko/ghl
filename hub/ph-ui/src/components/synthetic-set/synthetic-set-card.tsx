import { SyntheticDataset } from '@/types/synthetic-dataset'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { IconDownload, IconEdit, IconTrash } from '@tabler/icons-react'
import { formatDistanceToNow } from 'date-fns'

interface SyntheticSetCardProps {
  dataset: SyntheticDataset
  onViewDataset: (dataset: SyntheticDataset) => void
  onEditDataset: (dataset: SyntheticDataset) => void
  onDeleteDataset: (dataset: SyntheticDataset) => void
  onDownloadDataset: (dataset: SyntheticDataset) => void
}

export function SyntheticSetCard({
  dataset,
  onViewDataset,
  onEditDataset,
  onDeleteDataset,
  onDownloadDataset,
}: SyntheticSetCardProps) {
  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold">{dataset.name}</CardTitle>
        <CardDescription>
          {dataset.description || 'No description provided'}
        </CardDescription>
      </CardHeader>
      <CardContent className="pb-2">
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="text-muted-foreground">Patients:</div>
          <div className="font-medium">{dataset.num_patients}</div>
          
          <div className="text-muted-foreground">Created:</div>
          <div className="font-medium">
            {formatDistanceToNow(new Date(dataset.created_at), { addSuffix: true })}
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between pt-2">
        <Button variant="outline" size="sm" onClick={() => onViewDataset(dataset)}>
          View Details
        </Button>
        <div className="flex space-x-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onDownloadDataset(dataset)}
            title="Download Dataset"
          >
            <IconDownload size={18} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onEditDataset(dataset)}
            title="Edit Dataset"
          >
            <IconEdit size={18} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => onDeleteDataset(dataset)}
            title="Delete Dataset"
            className="text-destructive hover:text-destructive"
          >
            <IconTrash size={18} />
          </Button>
        </div>
      </CardFooter>
    </Card>
  )
}
