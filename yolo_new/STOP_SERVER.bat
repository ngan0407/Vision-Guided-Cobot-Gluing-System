@echo off
echo.
echo ===================================
echo  Cobot Server Shutdown Script
echo ===================================
echo.
echo Finding server process on port 5000...
echo.

set PID=
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5000" ^| findstr "LISTENING"') do (
    set PID=%%a
)

if "%PID%"=="" (
    echo ERROR: No active server found listening on port 5000.
    echo The server is already stopped.
) else (
    echo Server process found! (PID: %PID%)
    echo.
    echo Sending termination command...
    taskkill /F /PID %PID% > nul
    echo.
    echo Server has been successfully stopped.
)

echo.
echo ===================================
echo Closing window in 2 seconds...

:: This 'timeout' command acts as a 2-second pause
timeout /t 2 /nobreak > nul

:: The script ends here and the window will close