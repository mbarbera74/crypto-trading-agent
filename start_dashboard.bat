@echo off
title Trading Agent - Dashboard
echo ========================================
echo   Trading Agent - Avvio Dashboard
echo ========================================
echo.

cd /d "C:\Users\marcobarbera\Downloads\Github Projects\crypto_trading_agent"

:: Chiudi eventuali istanze precedenti sulla porta 8503
echo Chiusura istanze precedenti...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8503" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

echo Avvio dashboard su http://localhost:8503 ...
echo Premi Ctrl+C per chiudere.
echo.

"C:\Users\marcobarbera\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run dashboard/app.py --server.port 8503

pause
