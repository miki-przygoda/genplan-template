# genplan-template

A template repository for generative AI floor plan research and coursework.

## 🏗️ Project Structure

```
genplan-template/
├── dataset/                    # Downloaded datasets (auto-created)
│   ├── floor_plan_kaggle/     # G-list/floor_plan_kaggle
│   ├── FloorPlans970Dataset/  # HamzaWajid1/FloorPlans970Dataset
│   └── floorplan-SDXL/        # FahadIqbal5188/floorplan-SDXL
├── processed/                 # Your processed data (create as needed)
├── utils/
│   └── data_loader.py        # Automated dataset downloader
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Modern Python project configuration
└── README.md                 # This file
```

## 🚀 How It Works

### 1. **Automated Dataset Download**
The `utils/data_loader.py` script automatically:
- Downloads 4 pre-configured floor plan datasets from Hugging Face
- Organizes each dataset into its own folder
- Creates the `dataset/` directory structure
- Provides progress feedback and error handling

### 2. **Standardized Workflow**
- **Download**: Run the data loader to get all datasets
- **Process**: Work with datasets in the `processed/` folder
- **Maintain**: Keep consistent folder structure for team collaboration

### 3. **Team Collaboration**
- Each team member forks this repository
- Maintains the same folder structure for easy comparison
- Uses the same dataset organization system

## 🛠️ Setup Instructions

This project requires only the **Hugging Face Datasets** library and its core dependencies.

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

## 🎯 Getting Started

### Step 1: Download All Datasets
```bash
# Make sure your environment is activated
conda activate genai  # or source genai-env/bin/activate

# Run the data loader
python utils/data_loader.py
```

This will create the `dataset/` folder and download all 4 floor plan datasets.

### Step 2: Explore Your Data
```python
from datasets import load_dataset
import os

# List available datasets
dataset_folders = os.listdir("dataset")
print("Available datasets:", dataset_folders)

# Load a specific dataset
dataset = load_dataset("dataset/floor_plan_kaggle")
print(f"Dataset info: {dataset}")
print(f"Number of samples: {len(dataset['train'])}")
```

### Step 3: Start Your Research
- Create your processing scripts in the root directory
- Save processed data to the `processed/` folder
- Maintain the folder structure for team compatibility

## 📦 Dependencies

The project uses minimal dependencies focused on dataset handling:

| Package | Purpose |
|---------|---------|
| `datasets` | Main Hugging Face datasets library |
| `huggingface-hub` | For accessing Hugging Face Hub |
| `pandas` | Data manipulation and analysis |
| `pyarrow` | Fast data processing and storage |
| `numpy` | Numerical computing |

## 🤝 Team Workflow

1. **Fork this repository**
2. **Set up your environment** (see Setup Instructions)
3. **Download datasets** using `python utils/data_loader.py`
4. **Create your research scripts** while maintaining folder structure
5. **Save processed data** to the `processed/` folder
6. **Keep consistent naming** for easy team comparison

## 📁 Folder Guidelines

- **`dataset/`**: Downloaded datasets (auto-created, don't commit to git)
- **`processed/`**: Your processed/cleaned data
- **`utils/`**: Shared utility scripts
- **Root**: Your main research scripts

## 🔧 Troubleshooting

**Import errors in IDE:**
- Make sure your IDE is using the correct Python interpreter
- In VS Code/Cursor: `Cmd+Shift+P` → "Python: Select Interpreter" → Choose your conda/venv environment

**Dataset download issues:**
- Check your internet connection
- Ensure you have enough disk space
- Some datasets may be large (several GB)

**Environment issues:**
- Make sure you've activated your environment before running scripts
- Try `conda activate genai` or `source genai-env/bin/activate`