# Hybrid NIDS Dashboard

This is a read-only Streamlit dashboard for the existing Hybrid NIDS project.
It displays saved artifacts and reports, scores uploaded flow CSV files with
Engine A, and never retrains models when the page refreshes.

## Run from the project root

```powershell
& ".venv\Scripts\python.exe" -m pip install -r dashboard\requirements.txt
& ".venv\Scripts\python.exe" -m streamlit run dashboard\app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

For GPU execution, use a GPU-enabled WSL/Linux Python environment; native
Windows TensorFlow 2.21 generally runs on CPU.
