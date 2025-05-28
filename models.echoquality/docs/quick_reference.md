# EchoQuality Quick Reference

## Basic Usage

```bash
# Run inference on all DICOM files in raw_data/ directory
make inference

# View results
ls results/inference_output/
```

## Directory Flow

**Important**: Each folder in `raw_data/` represents one patient using one device in one study.

```
raw_data/ (input)                      →  preprocessed_data/ (extracted images)  →  results/ (analysis)
├── patient_001_device_A_study_001/        ├── patient_001_device_A_study_001/  ├── summary.json
│   ├── view1.dcm                          │   ├── view1_frame_00.png           ├── patient_001_device_A_study_001/
│   ├── view2.dcm                          │   ├── view1_frame_01.png           │   ├── folder_summary.json
│   └── view3.dcm                          │   └── ...                          │   ├── inference_results.json
├── patient_002_device_B_study_001/        ├── patient_002_device_B_study_001/  │   ├── *.png (charts)
│   ├── apical_4ch.dcm                     │   ├── apical_4ch_frame_00.png      │   └── failed_files/
│   └── parasternal_long.dcm               │   └── ...                          ├── patient_002_device_B_study_001/
└── ...                                    └── ...                              │   └── ...
                                                                                └── ...
```

## Key Output Files

| File | Description |
|------|-------------|
| `summary.json` | Overall statistics across all devices |
| `{device}/folder_summary.json` | Per-device summary |
| `{device}/inference_results.json` | Detailed quality scores per file |
| `{device}/{device}_failed_files.json` | Error details |
| `{device}/*.png` | Visualization charts |

## Quality Scores

| Score | Quality | Action |
|-------|---------|--------|
| ≥ 0.8 | Excellent | ✅ Use for analysis |
| 0.6-0.8 | Good | ✅ Generally acceptable |
| 0.3-0.6 | Acceptable | ⚠️ Review manually |
| < 0.3 | Poor/Fail | ❌ Exclude or reacquire |

## Common Commands

```bash
# Basic inference
make inference

# Custom threshold
poetry run python -m inference.inference --threshold 0.5

# Enable debug visualizations
poetry run python -m inference.inference --gradcam

# Process custom directory
poetry run python -m inference.inference --data_dir ./custom_raw
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No DICOM files found" | Check `raw_data/` directory exists with .dcm files |
| Model loading error | Verify `weights/video_quality_model.pt` exists |
| Permission error | Check write access to `preprocessed_data/` and `results/` |
| Memory error | Use `--device cpu` or reduce data size |

For detailed documentation, see [inference_pipeline.md](inference_pipeline.md).
