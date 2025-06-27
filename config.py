import os
from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
PLANTVILLAGE_DIR = DATA_DIR / "plantvillage"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
TRANSFER_MODEL_PATH = MODELS_DIR / "transfer_best_model.h5"
CUSTOM_MODEL_PATH = MODELS_DIR / "custom_best_model.h5"

# Results paths
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
def create_directories():
    """Create necessary directories"""
    directories = [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print("Directory structure created successfully!")

# Validate data directory
def validate_data_directory():
    """Check if data directory exists and has valid structure"""
    if not PLANTVILLAGE_DIR.exists():
        print(f"❌ Data directory not found: {PLANTVILLAGE_DIR}")
        print("📁 Expected directory structure:")
        print("data/")
        print("└── plantvillage/")
        print("    ├── Apple___Apple_scab/")
        print("    ├── Apple___Black_rot/")
        print("    └── ... (other plant disease folders)")
        return False
    
    # Check for subdirectories (classes)
    subdirs = [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir()]
    if len(subdirs) == 0:
        print(f"❌ No class directories found in: {PLANTVILLAGE_DIR}")
        return False
    
    print(f"✅ Found {len(subdirs)} class directories")
    return True

if __name__ == "__main__":
    create_directories()
    validate_data_directory()