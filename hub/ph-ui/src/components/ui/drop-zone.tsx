import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { cn } from '@/lib/utils'
import { UploadIcon } from 'lucide-react'

interface DropZoneProps {
  onDrop: (acceptedFiles: File[]) => void
  accept?: Record<string, string[]>
  maxFiles?: number
  maxSize?: number
  disabled?: boolean
  className?: string
  activeClassName?: string
  disabledClassName?: string
  children?: React.ReactNode
}

export function DropZone({
  onDrop,
  accept,
  maxFiles = 0,
  maxSize = 5 * 1024 * 1024 * 1024, // Default to 5GB max size
  disabled = false,
  className,
  activeClassName = 'border-primary',
  disabledClassName = 'opacity-50 cursor-not-allowed',
  children,
}: DropZoneProps) {
  const [error, setError] = useState<string | null>(null)

  const handleDrop = useCallback(
    (acceptedFiles: File[]) => {
      setError(null)
      onDrop(acceptedFiles)
    },
    [onDrop]
  )

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop: handleDrop,
    accept,
    maxFiles,
    maxSize,
    disabled,
    onDropRejected: (rejections) => {
      // Handle file rejection errors
      const errors = rejections.map((rejection) => {
        if (rejection.errors[0].code === 'file-too-large') {
          return `File ${rejection.file.name} is too large`
        }
        if (rejection.errors[0].code === 'file-invalid-type') {
          return `File ${rejection.file.name} has an invalid file type`
        }
        if (rejection.errors[0].code === 'too-many-files') {
          return `Too many files selected`
        }
        return rejection.errors[0].message
      })
      setError(errors[0])
    },
  })

  return (
    <div
      {...getRootProps()}
      className={cn(
        'flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/50 p-6 text-center transition-colors hover:bg-muted/80',
        isDragActive && activeClassName,
        disabled && disabledClassName,
        className
      )}
    >
      <input {...getInputProps()} />
      
      {children || (
        <>
          <UploadIcon className="mb-4 h-10 w-10 text-muted-foreground" />
          <div className="space-y-1">
            <p className="text-sm font-medium">
              Drag & drop files here, or click to select files
            </p>
            <p className="text-xs text-muted-foreground">
              {accept
                ? `Accepted file types: ${Object.values(accept)
                    .flat()
                    .join(', ')}`
                : 'All file types accepted'}
              {maxSize > 0 && ` (Max size: ${
                maxSize >= 1024 * 1024 * 1024
                  ? `${(maxSize / 1024 / 1024 / 1024).toFixed(1)}GB`
                  : `${(maxSize / 1024 / 1024).toFixed(1)}MB`
              })`}
              {maxFiles > 0 && ` (Max files: ${maxFiles})`}
            </p>
          </div>
        </>
      )}
      
      {error && (
        <div className="mt-4 text-sm font-medium text-destructive">{error}</div>
      )}
    </div>
  )
}
