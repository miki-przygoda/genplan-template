# Evolutionary Algorithms for Automated Floor Plan Generation

A research repository investigating the application of evolutionary algorithms (EAs) and reinforcement learning (RL) techniques for the automated generation and optimization of architectural floor plans. This work extends the base template by exploring population-based search strategies to generate novel floor plan layouts that satisfy spatial constraints derived from natural language descriptions.

## Abstract

*To be added: Comprehensive abstract describing the research objectives, methodology, and preliminary findings.*

## Motivation

Traditional floor plan generation relies heavily on manual design processes, which are time-consuming and may not efficiently explore the vast space of possible layouts. This project investigates whether evolutionary algorithms and reinforcement learning can effectively generate functional floor plans that adhere to spatial constraints specified in natural language, such as "The first bedroom is located at south west" or "The first bathroom is located at north east."

## Objectives

*To be added: Detailed research objectives and hypotheses.*

## Methodology

### Dataset

This project utilizes the **FloorPlans970Dataset** (HamzaWajid1/FloorPlans970Dataset), a collection of 970 floor plan images with associated natural language descriptions. Most samples includes:
- A 512×512 pixel color-coded floor plan image
- Supporting text describing room locations and relationships using compas directions

**Note:** A preliminary data audit identified 35 samples with missing supporting text, which are documented in `processed/no_text_ids.json`.

### Preprocessing Pipeline

The preprocessing stage transforms raw floor plan images and text descriptions into structured representations suitable for evolutionary search:

1. **Data Inspection** (`utils/pre_processing.py`): Validates image dimensions, identifies samples with missing text, and exports sample images for manual verification.
2. **Room Extraction**: Color-based contour detection to extract room polygons and bounding boxes from floor plan images.
3. **Text Parsing**: Natural language processing to extract spatial relationships (e.g., cardinal directions, zone assignments) from descriptive text.
4. **Orientation Normalization**: Automatic rotation detection and correction to standardize floor plan orientations.
5. **Grid Encoding**: Spatial discretization of floor plans into 2×2 and 4×4 grid representations, where each cell encodes room occupancy and type.

### Evolutionary Algorithm Framework

The core evolutionary search strategy employs a population-based approach with the following components:

- **Genome Representation**: Grid-encoded floor plan layouts, where individuals represent complete spatial configurations.
- **Fitness Function**: Multi-objective evaluation combining:
  - Zone constraint satisfaction (alignment with text-derived spatial requirements)
  - Overlap penalties (penalizing invalid room intersections)
  - Compactness measures (rewarding efficient space utilization)
- **Genetic Operators**: Standard EA operators including selection, crossover, and mutation adapted for spatial layout genomes.

See `TODO.md` for detailed implementation roadmap and current progress.

### Reinforcement Learning Integration

*To be added: Description of RL components and how they interact with the evolutionary framework.*

## Project Structure

```
genplan-template/
├── dataset/                      # Downloaded datasets (auto-created)
│   └── FloorPlans970Dataset/    # Primary dataset (970 floor plan samples)
├── processed/                    # Preprocessed data and intermediate results
│   ├── no_text_ids.json         # Identifiers for samples without text
│   ├── rooms/                   # Extracted room annotations (to be implemented)
│   └── encoded/                 # Grid-encoded layouts (to be implemented)
├── src/                         # Source code modules
│   └── evo_floorplan/           # Core evolutionary algorithm implementation
│       ├── extract_rooms_color.py  # Room extraction via color-based detection
│       ├── text_parser.py          # Natural language to structured metadata
│       ├── orient.py               # Orientation normalization
│       └── evolution.py            # EA framework (selection, crossover, mutation)
├── utils/                       # Utility scripts and preprocessing tools
│   ├── data_loader.py           # Automated dataset downloader
│   └── pre_processing.py        # Data inspection and validation
├── logs/                        # Evolutionary run logs (to be created)
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Modern Python project configuration
├── TODO.md                      # Implementation roadmap and task tracking
└── README.md                    # This file
```

## 🛠️ Setup Instructions

This project requires Python 3.11+ and standard scientific computing libraries.

### Option 1: Using Conda (Recommended)

1. **Create a new conda environment:**
   ```bash
   conda create -n genai python=3.11 -y
   ```

2. **Activate the environment:**
   ```bash
   conda activate genai
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Python venv

1. **Create a virtual environment:**
   ```bash
   python -m venv genai-env
   ```

2. **Activate the environment:**
   - On macOS/Linux:
     ```bash
     source genai-env/bin/activate
     ```
   - On Windows:
     ```bash
     genai-env\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

### Step 1: Download Dataset

```bash
# Make sure your environment is activated
conda activate genai  # or source genai-env/bin/activate

# Run the data loader
python utils/data_loader.py
```

This will download the FloorPlans970Dataset to `dataset/FloorPlans970Dataset/`.

### Step 2: Run Data Inspection

```bash
# Perform initial data audit
python utils/pre_processing.py --sample-count 20 --sample-output-dir processed/samples
```

This validates image dimensions, identifies samples with missing text, and exports sample images for manual inspection.

### Step 3: Begin Preprocessing

Follow the implementation roadmap in `TODO.md` to progressively build the preprocessing pipeline (room extraction, text parsing, grid encoding).

## Dependencies

The project uses the following core dependencies:

| Package | Purpose |
|---------|---------|
| `datasets` | Main Hugging Face datasets library |
| `huggingface-hub` | For accessing Hugging Face Hub |
| `pandas` | Data manipulation and analysis |
| `pyarrow` | Fast data processing and storage |
| `numpy` | Numerical computing |
| `Pillow` | Image processing utilities |

*Additional dependencies for evolutionary algorithms and reinforcement learning to be added as implementation progresses.*

## Implementation Status

Current progress is tracked in `TODO.md`. As of the latest update:

- ✅ **Data Inspection**: Initial audit complete; missing text samples identified
- 🚧 **Room Extraction**: Color-based detection in progress
- ⏳ **Text Parsing**: Planned
- ⏳ **Evolutionary Framework**: Planned
- ⏳ **Reinforcement Learning Integration**: Planned

## Experiments

*To be added: Experimental setup, hyperparameter configurations, and evaluation metrics.*

## Results

*To be added: Quantitative results, qualitative analysis, and comparative evaluations.*

## Discussion

*To be added: Analysis of results, limitations, and future research directions.*

## Contributions

*To be added: Statement of contributions and acknowledgments.*

## Citations

*To be added: References to related work, datasets, and foundational papers.*

## Folder Guidelines

- **`dataset/`**: Downloaded datasets (auto-created, do not commit to git)
- **`processed/`**: Preprocessed data, annotations, and intermediate results
- **`src/evo_floorplan/`**: Core evolutionary algorithm and preprocessing modules
- **`utils/`**: Utility scripts for data loading and inspection
- **`logs/`**: Evolutionary run logs and experiment tracking (to be created)

## Troubleshooting

**Import errors in IDE:**
- Ensure your IDE is using the correct Python interpreter
- In VS Code/Cursor: `Cmd+Shift+P` → "Python: Select Interpreter" → Choose your conda/venv environment

**Dataset download issues:**
- Verify internet connection and available disk space
- The FloorPlans970Dataset may require several GB of storage

**Environment issues:**
- Activate your environment before running scripts: `conda activate genai` or `source genai-env/bin/activate`

**Preprocessing errors:**
- Ensure the dataset has been downloaded: `python utils/data_loader.py`
- Check that image dimensions match expected 512×512 resolution
