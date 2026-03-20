@echo off
cd /d D:\ultimate

echo Starting Streamlit...
start cmd /k streamlit run app.py --server.headless true

timeout /t 5 >nul

echo Opening User Browser...
start chrome http://localhost:8501

timeout /t 2 >nul

echo Opening Admin Browser...
start chrome http://localhost:8501

exit