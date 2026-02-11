# master-thesis
For developing and solving the NMMWPBP

## Requirements

- Python 3.12.x
- Gurobi 12.0.x (must be installed separately)

## Setup

### macOS / Linux
python -m venv thesis-env
source thesis-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

### Windows
python -m venv thesis-env
thesis-env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

## Run

python -m scripts.run_main