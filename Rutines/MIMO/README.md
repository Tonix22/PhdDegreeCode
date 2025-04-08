# README for MIMO

## Overview
The `MimoWithChannel.sh` script automates the process of generating datasets, training machine learning models, and testing them. It integrates MATLAB and Python workflows to streamline the MIMO (Multiple Input Multiple Output) signal processing pipeline.

This script supports the following operations:
1. **Data Generation** (`-g`): Generates datasets using MATLAB.
2. **Training** (`-t`): Trains machine learning models using Python.
3. **Testing** (`-r`): Tests the trained models and visualizes results.

You can combine multiple operations in a single execution by using flags like `-gt` or `-gtr`.

---

## Prerequisites
Before running the script, ensure the following dependencies are installed and configured:

### MATLAB
- MATLAB must be installed and accessible via the command line.
- Ensure the `matlab` command is in your system's PATH.

### Python
- Python 3.x must be installed.
- Required Python libraries:
  - `pytorch-lightning`
  - `torch`
  - `numpy`
  - `scipy`
- Install dependencies using:
  ```bash
  pip install -r requirements.txt