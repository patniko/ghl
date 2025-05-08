# API Reference

## Data Modules

### `PTBXLDataset`
```python
PTBXLDataset(
    root_dir: str,
    sample_rate: int = 100,
    duration: float = 10.0,
    leads: List[str] = ['I'],
    augment: bool = False,
    metadata: bool = True,
    lazy_load: bool = False,
    strat_fold: Optional[int] = None
)
```
Loads PTB-XL ECG dataset with metadata support.

### `CustomECGDataset`
```python
CustomECGDataset(
    data_path: str,
    sample_rate: int = 100,
    duration: float = 10.0,
    student_leads: List[str] = ['I'],
    teacher_leads: List[str] = STANDARD_LEADS,
    augment: bool = False,
    metadata: bool = True,
    metadata_schema: Optional[Dict] = None
)
```
Loads custom ECG datasets for asymmetric learning.

## Models

### `ECGTransformer`
```python
ECGTransformer(
    in_channels: int = 1,
    embed_dim: int = 768,
    num_heads: int = 8,
    num_layers: int = 12,
    use_metadata: bool = False,
    metadata_dim: int = 64
)
```
Vision Transformer adapted for ECG signals.

### `DINOV2`
```python
DINOV2(
    student: nn.Module,
    teacher: nn.Module,
    student_dim: int = 256,
    teacher_dim: int = 768,
    out_dim: int = 65536
)
```
Asymmetric DINO V2 implementation.
