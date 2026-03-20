
@echo off
cd /d %~dp0
start cmd /k "python -m streamlit run app.py"
timeout /t 5 >nul
start http://localhost:8501
exit
