# master-thesis
For developing and solving the NMMWPBP

## Requirements

- Python 3.12.x
- Gurobi 12.0.x (must be installed separately)


## Setup

### macOS / Linux
<pre>
```bash
python -m venv thesis-env
source thesis-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
</pre>

### Windows
python -m venv thesis-env
thesis-env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


## Run

python -m scripts.run_main


## Project structure
<pre>
```text
master-thesis/
│
├── data/                   # Input data
│
├── results/                # Generated results (not tracked by git)
│
├── scripts/
│   └── run_main.py         # Entry point for running the model
│
├── src/                    # Core model code
│   ├── model.py
│   ├── tree.py
│   ├── read.py
│   └── utils.py
│
├── experiments/            # Experiment logic
│   ├── benchmark.py
│   └── rvmss.py
│
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
```
</pre>

