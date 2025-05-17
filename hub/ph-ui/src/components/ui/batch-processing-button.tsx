import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { useToast } from '@/hooks/use-toast'
import { 
  PlayIcon, 
  SquareIcon, 
  RefreshCwIcon, 
  CheckCircleIcon, 
  XCircleIcon,
  Loader2Icon
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import api from '@/services/api'

interface BatchProcessingButtonProps {
  batchId: number
  orgSlug?: string
  onStatusChange?: (status: string) => void
  initialStatus?: string
  className?: string
}

export function BatchProcessingButton({
  batchId,
  orgSlug,
  onStatusChange,
  initialStatus,
  className
}: BatchProcessingButtonProps) {
  const { toast } = useToast()
  const [status, setStatus] = useState<string>(initialStatus || 'unknown')
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [isProcessing, setIsProcessing] = useState<boolean>(false)
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null)

  // Function to get the current processing status
  const fetchProcessingStatus = async () => {
    try {
      const endpoint = orgSlug 
        ? `/batches/${orgSlug}/processing/status?batch_id=${batchId}` 
        : `/batches/processing/status?batch_id=${batchId}`
      
      const response = await api.get(endpoint)
      const newStatus = response.data.status
      
      if (newStatus !== status) {
        setStatus(newStatus)
        if (onStatusChange) {
          onStatusChange(newStatus)
        }
      }
      
      setIsProcessing(response.data.is_processing)
    } catch (error) {
      console.error('Error fetching processing status:', error)
    }
  }

  // Start polling for status when processing is active
  useEffect(() => {
    // Initial status check
    fetchProcessingStatus()
    
    // Clean up interval on unmount
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval)
      }
    }
  }, [batchId, orgSlug])

  // Update polling based on processing state
  useEffect(() => {
    if (isProcessing) {
      // Start polling every 2 seconds when processing
      const interval = setInterval(fetchProcessingStatus, 2000)
      setPollingInterval(interval)
      
      return () => clearInterval(interval)
    } else if (pollingInterval) {
      // Stop polling when not processing
      clearInterval(pollingInterval)
      setPollingInterval(null)
    }
  }, [isProcessing, batchId, orgSlug])

  // Start batch processing
  const startProcessing = async () => {
    setIsLoading(true)
    try {
      const endpoint = orgSlug 
        ? `/batches/${orgSlug}/processing/process?batch_id=${batchId}` 
        : `/batches/processing/process?batch_id=${batchId}`
      
      const response = await api.post(endpoint)
      
      if (response.data.status === 'already_processing') {
        toast({
          title: 'Already Processing',
          description: 'This batch is already being processed',
          variant: 'default',
        })
      } else {
        toast({
          title: 'Processing Started',
          description: 'Batch processing has been started',
          variant: 'default',
        })
        
        setStatus(response.data.status)
        setIsProcessing(true)
        
        if (onStatusChange) {
          onStatusChange(response.data.status)
        }
      }
    } catch (error) {
      console.error('Error starting batch processing:', error)
      toast({
        title: 'Error',
        description: 'Failed to start batch processing',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Cancel batch processing
  const cancelProcessing = async () => {
    setIsLoading(true)
    try {
      const endpoint = orgSlug 
        ? `/batches/${orgSlug}/processing/cancel?batch_id=${batchId}` 
        : `/batches/processing/cancel?batch_id=${batchId}`
      
      const response = await api.post(endpoint)
      
      toast({
        title: 'Processing Cancelled',
        description: 'Batch processing cancellation requested',
        variant: 'default',
      })
      
      if (onStatusChange) {
        onStatusChange('cancelling')
      }
    } catch (error) {
      console.error('Error cancelling batch processing:', error)
      toast({
        title: 'Error',
        description: 'Failed to cancel batch processing',
        variant: 'destructive',
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Get status badge color
  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'processing':
      case 'queued':
        return 'bg-blue-100 text-blue-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      case 'cancelled':
      case 'cancelling':
        return 'bg-orange-100 text-orange-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return <CheckCircleIcon className="h-4 w-4" />
      case 'failed':
        return <XCircleIcon className="h-4 w-4" />
      case 'processing':
      case 'queued':
        return <Loader2Icon className="h-4 w-4 animate-spin" />
      default:
        return null
    }
  }

  // Determine if the process button should be disabled
  const isProcessButtonDisabled = isLoading || ['processing', 'queued', 'cancelling'].includes(status.toLowerCase())

  // Determine if the cancel button should be shown
  const showCancelButton = ['processing', 'queued'].includes(status.toLowerCase())

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      {status !== 'unknown' && (
        <Badge className={getStatusColor(status)}>
          <span className="flex items-center gap-1">
            {getStatusIcon(status)}
            {status.toUpperCase()}
          </span>
        </Badge>
      )}
      
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="default"
              size="sm"
              onClick={startProcessing}
              disabled={isProcessButtonDisabled}
              className={showCancelButton ? 'hidden' : ''}
            >
              {isLoading ? (
                <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <PlayIcon className="mr-2 h-4 w-4" />
              )}
              Process
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Start processing this batch</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      
      {showCancelButton && (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="destructive"
                size="sm"
                onClick={cancelProcessing}
                disabled={isLoading || status.toLowerCase() === 'cancelling'}
              >
                {isLoading ? (
                  <Loader2Icon className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <SquareIcon className="mr-2 h-4 w-4" />
                )}
                Cancel
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Cancel the current processing job</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      )}
      
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              onClick={fetchProcessingStatus}
              disabled={isLoading}
            >
              <RefreshCwIcon className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Refresh status</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  )
}
