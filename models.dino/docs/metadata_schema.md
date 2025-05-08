# Metadata Schema Specification

## Standard Schema
```yaml
age:
  scale: 100.0  # Normalize by dividing age by 100
sex:
  categories: ['F', 'M']  # Encode as 0/1
height:
  scale: 200.0  # Normalize height by 200cm
weight:
  scale: 150.0  # Normalize weight by 150kg
```

## Custom Schema Example
```python
{
    'patient_age': {
        'scale': 120.0  # Normalize age by 120 years
    },
    'gender': {
        'categories': ['Male', 'Female', 'Other']  # Encoded as 0,1,2
    },
    'bmi': {
        'scale': 50.0  # Normalize BMI by 50
    }
}
```

## Usage
```python
from data.custom_dataset import CustomECGDataset

custom_schema = {
    'age': {'scale': 120.0},
    'gender': {'categories': ['M', 'F']}
}

dataset = CustomECGDataset(
    data_path='/path/to/data',
    metadata_schema=custom_schema
)
```
