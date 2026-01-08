# BMAN73701 – Python Programming for Business Intelligence and Analytics  
## Coursework Project (2025–26)

This project is a **Streamlit-based interactive application** developed for the BMAN73701 coursework.  
It implements **Tasks 1–8** as specified in the coursework brief, combining optimisation, analytics, and data management.

---

##  How to Run the App
### 1. Install dependencies
```bash
pip install -r requirements.txt


###
1.Install dependencies
bash
pip install -r requirements.txt

2. Run Streamlit
streamlit run app.py


Project Structure
.
├── app.py                  # Main Streamlit app
├── README.md               # Project description (this file)
├── requirements.txt        # Python dependencies
├── data/                   # Input data (CSV)
├── outputs/                # Generated outputs (plots, logs)
└── src/
    ├── task1_3.py           # Tasks 1–3: Scheduling optimisation
    ├── task4.py             # Task 4
    ├── task5.py             # Task 5
    ├── task6.py             # Task 6
    └── task8_data.py        # Task 8: CRUD + Logging

Tasks Overview
Task 1 – Cost Minimisation
Baseline staff scheduling using linear programming.
Task 2 – Fairness Constraints
Two fairness scenarios compared against the baseline.
Task 3 – Skills Constraints
Additional skill coverage constraints applied to the schedule.
Task 4–6 – Analytics & Visualisation
Exploratory analysis, metrics, and model-based evaluation.
Task 8 – Data Management (CRUD) + Logging
Interactive data management interface with:
Retrieve
Range filter
Delete
Modify
Audit log (JSONL → table)
All user actions are logged for auditability.

Technologies Used
Python
Streamlit
Pandas
PuLP (Linear Programming)
Matplotlib / Seaborn

Notes
__pycache__/ and .pyc files are excluded via .gitignore
Logs are generated dynamically in outputs/logs/

Author
Shuaijie Wang
MSc Business Analysis and Strategic Management
University of Manchester
