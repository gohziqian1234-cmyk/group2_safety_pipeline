@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo EG2A17 Group 5 - NESSO Live Safety System
echo ============================================================

echo Starting BLE gateway...
start "NESSO BLE Gateway" cmd /k py src\21_live_ble_gateway.py

timeout /t 2 /nobreak >nul

echo Starting Streamlit dashboard...
start "NESSO Dashboard" cmd /k py -m streamlit run src\dashboard.py

echo.
echo Two windows were opened:
echo   1. BLE gateway
echo   2. Streamlit dashboard
echo.
echo Confirm the gateway shows CONNECTED / SAFE before the lesson test.
pause
