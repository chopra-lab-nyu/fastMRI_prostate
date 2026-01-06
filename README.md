# FastMRI Prostate

[[`Paper`](https://www.nature.com/articles/s41597-024-03252-w)] [[`Dataset`](https://fastmri.med.nyu.edu/)] [[`Github`](https://github.com/cai2r/fastMRI_prostate)] [[`BibTeX`](#cite)]

### Updates
01-06-2026: Added work-stealing parallelization for streaming workers, `--skip-kspace` flag to reduce HDF5 output size, and improved documentation.

02-07-2024: Updated [files](https://github.com/cai2r/fastMRI_prostate/pull/11) for slice-, volume-, exam-level labels and their paths for T2 and Diffusion sequences in the [fastMRI prostate dataset](https://fastmri.med.nyu.edu/).

[Classification](https://github.com/cai2r/fastMRI_prostate/tree/main/fastmri_prostate_classification): The classification folder contains code for training deep learning models to detect clinically significant prostate cancer.
[Reconstruction](https://github.com/cai2r/fastMRI_prostate/tree/main/DL_reconstruction): The reconstruction folder contains code for training deep learning models for reconstructing diffusion MRI images from undersampled k-space.

## Overview

This repository contains code to facilitate the reconstruction of prostate T2 and DWI (Diffusion-Weighted Imaging) images from raw (k-space) data from the fastMRI Prostate dataset. It includes reconstruction methods along with utilities for pre-processing and post-processing the data. 

The package is intended to serve as a starting point for those who want to experiment and develop alternate reconstruction techniques. 

## Installation

The code requires `python >= 3.9`

Install FastMRI Prostate: clone the repository locally and install with

```bash
git clone https://github.com/cai2r/fastMRI_prostate.git
cd fastmri_prostate
pip install -e .
```

### Dependencies

Core dependencies:
- `numpy`, `scipy`, `scikit-image`
- `h5py` - HDF5 file I/O
- `twixtools` - Siemens `.dat` file parsing (`pip install twixtools`)
- `torch` - Used for ESC optimization
- `pyyaml` - Configuration file parsing

For streaming pipeline:
- `pandas` - Manifest CSV processing

## Quick Start

### From fastMRI Dataset (HDF5 files)

```bash
python fastmri_prostate_recon.py \
    --data_path <path to dataset> \
    --output_path <path to store recons> \
    --sequence <t2/dwi/both>
```

### From Siemens .dat Files

```bash
python fastmri_prostate_recon_from_dat.py \
    --data-dir <directory with .dat files> \
    --output-dir <path to store recons> \
    --combines rss,espirit \
    --skip-kspace
```

## Package Structure

```
fastmri_prostate/
├── data/
│   └── mri_data.py              # Data loading utilities (HDF5, .dat files)
├── reconstruction/
│   ├── grappa.py                # GRAPPA parallel imaging reconstruction
│   ├── utils.py                 # FFT, cropping, flipping utilities
│   ├── t2/                      # T2-weighted reconstruction
│   └── dwi/                     # Diffusion-weighted reconstruction
│       ├── prostate_dwi_recon.py    # Main DWI reconstruction pipeline
│       ├── coil_combine.py          # ESPIRiT and coil combination
│       ├── diffusion_metrics.py     # ADC, trace, b1500 computation
│       └── regridding.py            # EPI trajectory correction
└── visualization/               # Plotting utilities
```

## DWI Reconstruction Pipeline

The DWI reconstruction pipeline (`fastmri_prostate_recon_from_dat.py`) performs:

1. **Trapezoidal regridding** - Corrects for EPI readout trajectory
2. **GRAPPA reconstruction** - Fills missing k-space lines using calibration data
3. **Coil combination** - Multiple methods available:
   - `rss` - Root sum-of-squares (fast, no calibration needed)
   - `espirit` - ESPIRiT sensitivity maps (better SNR, requires calibration)
   - `esc` - Emulated single coil (optimized weighted combination)
4. **Diffusion averaging** - Combines multiple averages per b-value/direction
5. **Metric computation** - ADC maps, trace images, synthetic b1500

### Command Line Options

```bash
python fastmri_prostate_recon_from_dat.py \
    --data-dir <input directory> \
    --output-dir <output directory> \
    --directions b50x,b50y,b50z,b1000x,b1000y,b1000z \
    --combines rss,espirit \
    --skip-metrics \
    --skip-kspace \
    --enable-phasecorr \
    --max-files 10 \
    --job-index 0 \
    --job-count 1
```

| Flag | Description |
|------|-------------|
| `--directions` | Comma-separated diffusion directions (default: all 6) |
| `--combines` | Coil combination methods: `rss`, `espirit`, `esc` |
| `--skip-metrics` | Skip ADC/trace/b1500 computation |
| `--skip-kspace` | Don't store k-space in output (saves ~90% disk space) |
| `--enable-phasecorr` | Apply odd/even EPI phase correction |
| `--max-files` | Limit number of files to process |
| `--job-index/count` | For parallel processing across multiple jobs |

### Output HDF5 Structure

```
output.h5
├── metadata/
│   ├── directions          # List of diffusion directions
│   ├── averaging_schemes   # Averaging configurations
│   └── combines            # Coil combination methods used
├── images/
│   ├── rss/
│   │   ├── per_average/{direction}     # Per-average images
│   │   └── b50_4_b1000_12/{direction}  # Averaged images
│   └── espirit/
│       └── ...
├── metrics/
│   └── {combine}/{scheme}/
│       ├── adc_map
│       ├── b1500
│       ├── trace_b50
│       └── trace_b1000
└── kspace/                 # Only if --skip-kspace not set
    └── post_grappa_full
```

### Output Size Estimates

| Configuration | Size per scan |
|--------------|---------------|
| Full output (with k-space) | ~20 GB |
| `--skip-kspace` | ~0.7 GB |

## Streaming Pipeline (HPC Clusters)

For processing large volumes of `.dat` files on HPC clusters with SLURM, use the streaming pipeline which coordinates file transfer and parallel reconstruction.

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Research Drive │────▶│  Staging Dir     │────▶│  Output Dir     │
│  (source_root)  │     │  (.dat + .ready) │     │  (.h5 files)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │
   transfer.py              worker.py (×N)
   (data_mover)             (CPU nodes)
```

### Configuration (`config/streaming.yaml`)

```yaml
transfer:
  manifest_csv: /path/to/file_list.csv    # CSV with 'path' and 'size' columns
  source_root: /mnt/research_drive        # Mount point for source files
  staging_dir: /scratch/staging           # Temporary staging directory
  max_staging_gb: 500                     # Throttle when staging exceeds this
  min_bytes: 1000000000                   # Skip files smaller than 1GB
  poll_seconds: 30                        # Transfer polling interval

process:
  output_dir: /scratch/output             # Final HDF5 output location
  directions: "all"                       # Or list: ["b50x", "b1000x", ...]
  averages: "all"                         # Or pair: [4, 12] for b50/b1000
  combines: ["rss", "espirit"]            # Coil combination methods
  skip_metrics: false                     # Compute ADC/trace/b1500
  skip_kspace: true                       # Don't store k-space (recommended)
  enable_phasecorr: false                 # EPI phase correction
  poll_seconds: 10                        # Worker polling interval
  delete_dat: true                        # Delete .dat after successful recon
```

### Running the Pipeline

**Step 1: Start workers first** (they'll wait for files)
```bash
sbatch sbatch/stream_workers.sh
```

**Step 2: Start transfer**
```bash
sbatch sbatch/stream_transfer.sh
```

### How It Works

**Transfer Script (`scripts/transfer.py`):**
1. Reads manifest CSV and filters by size
2. Parses Siemens filename metadata to pair main scans with prescans
3. Copies files to staging, creates `.ready` marker after each copy
4. Maintains `.copied_manifest.json` to track progress (resumable)
5. Throttles when staging exceeds `max_staging_gb`
6. Sets `.transfer_active` flag while running

**Worker Script (`scripts/worker.py`):**
1. Watches for `*.dat.ready` markers in staging
2. Uses **work-stealing**: any worker can grab any unlocked file
3. Claims files via atomic `mkdir()` lock (prevents duplicate processing)
4. Processes file → writes `.h5` → deletes `.dat` and markers
5. Exits when transfer inactive and no files remain

### Worker Coordination

Workers use a work-stealing pattern for optimal load balancing:
- No static assignment - any worker processes any available file
- Atomic locking via `mkdir()` prevents race conditions
- Random shuffling reduces lock contention
- Workers stay busy as long as work is available

### SLURM Resource Recommendations

| Coil Combines | Memory | CPUs | Notes |
|---------------|--------|------|-------|
| RSS only | 32G | 4 | Fastest, minimal memory |
| RSS + ESPIRiT | 48G | 6 | ESPIRiT SVD is memory-intensive |
| RSS + ESPIRiT + ESC | 64G | 8 | ESC optimization adds overhead |

### Monitoring

```bash
# Check worker status
squeue -u $USER | grep dwi

# Watch staging directory
watch -n 5 'ls /scratch/staging/*.ready 2>/dev/null | wc -l'

# Tail worker logs
tail -f logs_streaming/dwi_stream_*.err

# Count completed reconstructions
ls /scratch/output/*.h5 | wc -l
```

### Troubleshooting

**Workers idle but files exist:**
- Check for `.ready` markers: `ls staging/*.ready`
- If missing, create them: `for f in staging/*.dat; do touch "${f}.ready"; done`

**Resume after failure:**
- Transfer automatically skips files in `.copied_manifest.json`
- To reprocess: remove file from manifest or delete manifest

**Clear staging for fresh start:**
```bash
rm staging/*.dat staging/*.ready staging/*.failed
echo '{"files": []}' > staging/.copied_manifest.json
```

## Hardware Requirements

- **Memory:** 32-64 GB RAM depending on coil combination methods
- **CPU:** Multi-core recommended (reconstruction is CPU-bound)
- **Storage:** ~0.7 GB per scan with `--skip-kspace`, ~20 GB without

### Runtime

| Sequence | Time per scan |
|----------|---------------|
| T2 | ~15 minutes |
| DWI (RSS only) | ~5 minutes |
| DWI (RSS + ESPIRiT) | ~7 minutes |
| DWI (RSS + ESPIRiT + ESC) | ~10 minutes |

## License
fastMRI_prostate is MIT licensed, as found in [LICENSE file](https://github.com/cai2r/fastMRI_prostate/blob/main/LICENSE)

## Cite
If you use the fastMRI Prostate data or code in your research, please use the following BibTeX entry.

```bibtex
@article{tibrewala2024fastmri,
  title={FastMRI Prostate: A public, biparametric MRI dataset to advance machine learning for prostate cancer imaging},
  author={Tibrewala, Radhika and Dutt, Tarun and Tong, Angela and Ginocchio, Luke and Lattanzi, Riccardo and Keerthivasan, Mahesh B and Baete, Steven H and Chopra, Sumit and Lui, Yvonne W and Sodickson, Daniel K and others},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={404},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgements
The code for the GRAPPA technique was based off [pygrappa](https://github.com/mckib2/pygrappa), and ESPIRiT maps provided in the dataset were computed using [espirit-python](https://github.com/mikgroup/espirit-python) 
