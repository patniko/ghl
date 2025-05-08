import React from 'react'
import { zodResolver } from '@hookform/resolvers/zod'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { DataRegion, ProjectCreate } from '@/types/project'
import { IconDatabase, IconMapPin } from '@tabler/icons-react'
import { cn } from '@/lib/utils'

// Form schema
const formSchema = z.object({
  name: z.string().min(1, 'Project name is required'),
  description: z.string().optional(),
  data_region: z.string().default(DataRegion.LOCAL),
  s3_bucket_name: z.string().optional(),
})

interface CreateProjectModalProps {
  isOpen: boolean
  onClose: () => void
  onCreateProject: (project: ProjectCreate) => void
  isLoading: boolean
}

export function CreateProjectModal({
  isOpen,
  onClose,
  onCreateProject,
  isLoading,
}: CreateProjectModalProps) {
  // Define available regions
  const regions = [
    {
      id: DataRegion.LOCAL,
      name: 'Local Storage',
      description: 'Store data locally on the server',
      icon: <IconDatabase className="h-6 w-6" />,
      available: true,
    },
    {
      id: DataRegion.INDIA,
      name: 'India',
      description: 'Store data in our India data center',
      icon: <IconMapPin className="h-6 w-6" />,
      available: true,
    },
    {
      id: DataRegion.US,
      name: 'United States',
      description: 'Store data in our US data center',
      icon: <IconMapPin className="h-6 w-6" />,
      available: true,
    },
  ]

  // Initialize form
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      name: '',
      description: '',
      data_region: DataRegion.LOCAL,
      s3_bucket_name: '',
    },
  })

  // Handle form submission
  const onSubmit = (values: z.infer<typeof formSchema>) => {
    onCreateProject({
      name: values.name,
      description: values.description,
      data_region: values.data_region,
      s3_bucket_name: values.s3_bucket_name,
    })
  }

  // Reset form when modal closes
  const handleClose = () => {
    form.reset()
    onClose()
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[550px]">
        <DialogHeader>
          <DialogTitle>Create New Project</DialogTitle>
          <DialogDescription>
            Create a new project to organize your files and data
          </DialogDescription>
        </DialogHeader>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Project Name</FormLabel>
                  <FormControl>
                    <Input placeholder="Enter project name" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description (Optional)</FormLabel>
                  <FormControl>
                    <Textarea
                      placeholder="Enter project description"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="space-y-3">
              <FormLabel>Data Region</FormLabel>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                {regions.map((region) => (
                  <FormField
                    key={region.id}
                    control={form.control}
                    name="data_region"
                    render={({ field }) => (
                      <FormItem className="space-y-0">
                        <FormControl>
                          <div
                            className={cn(
                              'flex cursor-pointer flex-col rounded-lg border p-4 transition-all hover:border-primary',
                              field.value === region.id
                                ? 'border-2 border-primary bg-primary/5'
                                : 'border-border',
                              !region.available && 'opacity-50'
                            )}
                            onClick={() => {
                              if (region.available) {
                                field.onChange(region.id)
                              }
                            }}
                          >
                            <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
                              {region.icon}
                            </div>
                            <div className="font-medium">{region.name}</div>
                            <div className="mt-1 text-xs text-muted-foreground">
                              {region.description}
                            </div>
                            {!region.available && (
                              <div className="mt-2 text-xs font-medium text-amber-500">
                                Coming soon
                              </div>
                            )}
                          </div>
                        </FormControl>
                      </FormItem>
                    )}
                  />
                ))}
              </div>
            </div>

            <FormField
              control={form.control}
              name="s3_bucket_name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Custom S3 Bucket Name (Optional)</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="Enter custom bucket name"
                      {...field}
                      value={field.value || ''}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={handleClose}
                disabled={isLoading}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isLoading}>
                {isLoading ? 'Creating...' : 'Create Project'}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}
