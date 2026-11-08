$ErrorActionPreference = 'Stop'
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe steer.py --experiment collagen --model facebook/esm2_t12_35M_UR50D --output results/collagen_35m
.\.venv\Scripts\python.exe plot.py results/collagen_35m/metrics.csv
