@echo off
cd /d D:\ultimate

echo =====================================
echo Starting Fintech Loan System...
echo =====================================

REM --- Try running Streamlit using Python module (works even if streamlit not in PATH)
start cmd /k python -m streamlit run app.py --server.headless true

REM --- Wait for server to start
timeout /t 6 >nul

echo Opening User Browser...
start "" http://localhost:8501

timeout /t 2 >nul

echo Opening Admin Browser...
start "" http://localhost:8501

exit
