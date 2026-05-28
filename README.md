# DeepFace Recognition Demo

This project demonstrates facial recognition and similarity comparison using the [DeepFace](https://github.com/serengil/deepface) library in Python.

## What This Script Does
The `deepface_demo.py` script performs the following tasks:
1. **Builds a Face Database**: Scans the `assets/contributors/` directory for images, detects faces using the `RetinaFace` detector, and extracts facial embeddings using the `Facenet` model.
2. **Analyzes a Selfie**: Detects and extracts facial embeddings from a target test image (selfie) located in the `assets/selfie/` directory.
3. **Face Matching**: Compares the selfie's facial embeddings against the known database to find matches based on cosine similarity. It then prints out a summary of any matches found along with their similarity percentages.

## Project Structure
- `deepface_demo.py`: The main Python script.
- `assets/contributors/`: Directory containing images of known people (the database). Add your reference images here.
- `assets/selfie/`: Directory containing the target test image(s) you want to recognize.

## Getting Started

### Prerequisites
Make sure you have Python installed on your system.

### 1. Setup a Virtual Environment (Recommended)
Open your terminal/command prompt in the project directory and run:
```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install deepface opencv-python numpy tf-keras
```

### 3. Run the Script
Execute the main script:
```bash
python deepface_demo.py
```

*Note: The first time you run the script, it will download the pre-trained weights for the `Facenet` and `RetinaFace` models. This might take a few moments depending on your internet connection.*

---