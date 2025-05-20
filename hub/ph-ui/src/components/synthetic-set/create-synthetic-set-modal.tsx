import { useState } from 'react'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Checkbox } from '@/components/ui/checkbox'
import { SyntheticDatasetCreate, SamplesDatasetCreate } from '@/types/synthetic-dataset'
import { IconUpload } from '@tabler/icons-react'
import { useToast } from '@/hooks/use-toast'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'

interface CreateSyntheticSetModalProps {
  isOpen: boolean
  onClose: () => void
  onCreateDataset: (dataset: SyntheticDatasetCreate, file?: File) => Promise<void>
  onCreateSamplesDataset?: (dataset: SamplesDatasetCreate) => Promise<void>
  isLoading: boolean
}

export function CreateSyntheticSetModal({
  isOpen,
  onClose,
  onCreateDataset,
  onCreateSamplesDataset,
  isLoading,
}: CreateSyntheticSetModalProps) {
  const { toast } = useToast()
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [numPatients, setNumPatients] = useState(10)
  const [file, setFile] = useState<File | null>(null)
  const [activeTab, setActiveTab] = useState('synthetic')
  const [uploadMode, setUploadMode] = useState(false)
  
  // Samples dataset specific state
  const [dataTypes, setDataTypes] = useState<string[]>(['questionnaire', 'blood', 'mobile', 'consent'])
  const [includePartials, setIncludePartials] = useState(false)
  const [partialRate, setPartialRate] = useState(0.3)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!name) {
      toast({
        title: 'Error',
        description: 'Please enter a name for the dataset',
        variant: 'destructive',
      })
      return
    }
    
    if (activeTab === 'synthetic') {
      if (!uploadMode && (numPatients < 1 || numPatients > 1000)) {
        toast({
          title: 'Error',
          description: 'Number of patients must be between 1 and 1000',
          variant: 'destructive',
        })
        return
      }
      
      if (uploadMode && !file) {
        toast({
          title: 'Error',
          description: 'Please select a CSV file to upload',
          variant: 'destructive',
        })
        return
      }
      
      const dataset: SyntheticDatasetCreate = {
        name,
        description: description || undefined,
        num_patients: uploadMode ? 0 : numPatients,
      }
      
      try {
        await onCreateDataset(dataset, uploadMode ? file! : undefined)
        resetForm()
      } catch (error) {
        console.error('Error creating dataset:', error)
      }
    } else if (activeTab === 'samples' && onCreateSamplesDataset) {
      if (numPatients < 1 || numPatients > 1000) {
        toast({
          title: 'Error',
          description: 'Number of patients must be between 1 and 1000',
          variant: 'destructive',
        })
        return
      }
      
      if (dataTypes.length === 0) {
        toast({
          title: 'Error',
          description: 'Please select at least one data type',
          variant: 'destructive',
        })
        return
      }
      
      const dataset: SamplesDatasetCreate = {
        name,
        description: description || undefined,
        num_patients: numPatients,
        data_types: dataTypes,
        include_partials: includePartials,
        partial_rate: partialRate,
      }
      
      try {
        await onCreateSamplesDataset(dataset)
        resetForm()
      } catch (error) {
        console.error('Error creating samples dataset:', error)
      }
    }
  }
  
  const resetForm = () => {
    setName('')
    setDescription('')
    setNumPatients(10)
    setFile(null)
    setUploadMode(false)
    setActiveTab('synthetic')
    setDataTypes(['questionnaire', 'blood', 'mobile', 'consent'])
    setIncludePartials(false)
    setPartialRate(0.3)
  }
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      if (selectedFile.type !== 'text/csv' && !selectedFile.name.endsWith('.csv')) {
        toast({
          title: 'Invalid file type',
          description: 'Please select a CSV file',
          variant: 'destructive',
        })
        return
      }
      setFile(selectedFile)
    }
  }
  
  const handleDataTypeChange = (type: string, checked: boolean) => {
    if (checked) {
      setDataTypes(prev => [...prev, type])
    } else {
      setDataTypes(prev => prev.filter(t => t !== type))
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Create Synthetic Dataset</DialogTitle>
          <DialogDescription>
            Create a new synthetic dataset with randomly generated data.
          </DialogDescription>
        </DialogHeader>
        
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="synthetic">Standard Synthetic Data</TabsTrigger>
            <TabsTrigger value="samples" disabled={!onCreateSamplesDataset}>GHL Samples Tool</TabsTrigger>
          </TabsList>
          
          <TabsContent value="synthetic" className="mt-4">
            <div className="text-sm text-muted-foreground mb-4">
              Generate synthetic patient data with a standard schema or upload your own CSV file.
            </div>
          </TabsContent>
          
          <TabsContent value="samples" className="mt-4">
            <div className="text-sm text-muted-foreground mb-4">
              Generate synthetic data using the GHL samples tool with multiple data types.
            </div>
          </TabsContent>
        </Tabs>
        
        <form onSubmit={handleSubmit} className="space-y-4 py-4">
          <div className="space-y-2">
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Enter dataset name"
              required
            />
          </div>
          
          <div className="space-y-2">
            <Label htmlFor="description">Description (optional)</Label>
            <Textarea
              id="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Enter dataset description"
              rows={3}
            />
          </div>
          
          {activeTab === 'synthetic' && (
            <>
              <div className="flex items-center space-x-2">
                <Button
                  type="button"
                  variant={uploadMode ? "outline" : "default"}
                  onClick={() => setUploadMode(false)}
                  className="flex-1"
                >
                  Generate Data
                </Button>
                <Button
                  type="button"
                  variant={uploadMode ? "default" : "outline"}
                  onClick={() => setUploadMode(true)}
                  className="flex-1"
                >
                  Upload CSV
                </Button>
              </div>
              
              {uploadMode ? (
                <div className="space-y-2">
                  <Label htmlFor="file">CSV File</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      id="file"
                      type="file"
                      accept=".csv"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => document.getElementById('file')?.click()}
                      className="w-full"
                    >
                      <IconUpload className="mr-2 h-4 w-4" />
                      {file ? file.name : 'Select CSV File'}
                    </Button>
                  </div>
                  {file && (
                    <p className="text-sm text-muted-foreground">
                      Selected file: {file.name} ({(file.size / 1024).toFixed(2)} KB)
                    </p>
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <Label htmlFor="numPatients">Number of Patients</Label>
                  <Input
                    id="numPatients"
                    type="number"
                    min={1}
                    max={1000}
                    value={numPatients}
                    onChange={(e) => setNumPatients(parseInt(e.target.value) || 10)}
                  />
                  <p className="text-sm text-muted-foreground">
                    Generate between 1 and 1000 synthetic patient records
                  </p>
                </div>
              )}
            </>
          )}
          
          {activeTab === 'samples' && (
            <>
              <div className="space-y-2">
                <Label htmlFor="samplesNumPatients">Number of Patients</Label>
                <Input
                  id="samplesNumPatients"
                  type="number"
                  min={1}
                  max={1000}
                  value={numPatients}
                  onChange={(e) => setNumPatients(parseInt(e.target.value) || 10)}
                />
                <p className="text-sm text-muted-foreground">
                  Generate between 1 and 1000 synthetic patient records
                </p>
              </div>
              
              <div className="space-y-2">
                <Label>Data Types</Label>
                <div className="grid grid-cols-2 gap-2">
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="questionnaire" 
                      checked={dataTypes.includes('questionnaire')}
                      onCheckedChange={(checked) => handleDataTypeChange('questionnaire', checked as boolean)}
                    />
                    <Label htmlFor="questionnaire" className="cursor-pointer">Questionnaire</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="blood" 
                      checked={dataTypes.includes('blood')}
                      onCheckedChange={(checked) => handleDataTypeChange('blood', checked as boolean)}
                    />
                    <Label htmlFor="blood" className="cursor-pointer">Blood Results</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="mobile" 
                      checked={dataTypes.includes('mobile')}
                      onCheckedChange={(checked) => handleDataTypeChange('mobile', checked as boolean)}
                    />
                    <Label htmlFor="mobile" className="cursor-pointer">Mobile Measures</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="consent" 
                      checked={dataTypes.includes('consent')}
                      onCheckedChange={(checked) => handleDataTypeChange('consent', checked as boolean)}
                    />
                    <Label htmlFor="consent" className="cursor-pointer">Consent</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="echo" 
                      checked={dataTypes.includes('echo')}
                      onCheckedChange={(checked) => handleDataTypeChange('echo', checked as boolean)}
                    />
                    <Label htmlFor="echo" className="cursor-pointer">Echo</Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Checkbox 
                      id="ecg" 
                      checked={dataTypes.includes('ecg')}
                      onCheckedChange={(checked) => handleDataTypeChange('ecg', checked as boolean)}
                    />
                    <Label htmlFor="ecg" className="cursor-pointer">ECG</Label>
                  </div>
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Checkbox 
                    id="includePartials" 
                    checked={includePartials}
                    onCheckedChange={(checked) => setIncludePartials(checked as boolean)}
                  />
                  <Label htmlFor="includePartials" className="cursor-pointer">Include Partial Records</Label>
                </div>
                <p className="text-sm text-muted-foreground">
                  Simulate real-world data by including patients with incomplete records
                </p>
              </div>
              
              {includePartials && (
                <div className="space-y-2">
                  <Label htmlFor="partialRate">Partial Rate</Label>
                  <div className="flex items-center gap-2">
                    <Input
                      id="partialRate"
                      type="range"
                      min={0}
                      max={1}
                      step={0.1}
                      value={partialRate}
                      onChange={(e) => setPartialRate(parseFloat(e.target.value))}
                      className="w-full"
                    />
                    <span className="w-12 text-center">{(partialRate * 100).toFixed(0)}%</span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Percentage of patients with partial records
                  </p>
                </div>
              )}
            </>
          )}
          
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose} disabled={isLoading}>
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading}>
              {isLoading ? 'Creating...' : 'Create Dataset'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
