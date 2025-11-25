@REM @echo off
@REM echo Starting the Cobot Alignment Server...

@REM :: This tells the script to run from the main project folder
@REM cd /d "E:\yolo_new\my_model"

@REM :: Start the server. 'start' runs it in a new window.
@REM :: We use 'python' so you can see any errors.
@REM start "Cobot Server" python web_server/app.py

@REM echo Server is starting up. Waiting 8 seconds...

@REM :: Wait for 7 seconds for the server to initialize
@REM timeout /t 8 /nobreak > nul

@REM echo Opening the web interface...

@REM :: Open the website in your default browser
@REM start http://127.0.0.1:5000

@REM exit


@echo off
echo Starting the Cobot Alignment Server...

:: This tells the script to run from the main project folder
cd /d "E:\yolo_new\my_model"

:: Start the server in its own window.
:: The logs will appear in this new "Cobot Server" window.
start "Cobot Server" python web_server/app.py

echo Waiting for server to respond at http://127.0.0.1:5000 ...
echo (This window will launch the browser automatically when ready)

:: This is the check loop
:check_server
:: Use curl (built-in to Windows 10/11) to ping the server.
:: -s = silent, -o NUL = discard output, -I = get headers only (faster)
:: --fail makes curl return an error (errorlevel non-zero) if the server fails (e.g., 404, 500, or connection refused)
curl.exe -s -o NUL -I --fail http://127.0.0.1:5000

:: %errorlevel% is 0 if the server responded successfully.
if %errorlevel% == 0 (
    goto :launch_browser
)

:: Server not ready, wait 1 second and check again
echo   - Server not ready. Retrying in 1 second...
timeout /t 1 /nobreak > nul
goto :check_server

:launch_browser
echo Server is running! Opening the web interface...
start http://127.0.0.1:5000

exit