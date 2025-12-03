@echo off
setlocal EnableDelayedExpansion

:: Check if Node.js is installed
where node >nul 2>&1
if %errorlevel%==0 (
    echo Node.js is installed: 
    node -v
    goto :run_app
)

echo Node.js not found. Downloading and installing...
:: Download latest Node.js LTS x64
set "NODE_URL=https://nodejs.org/dist/latest-v20.x/node-v20.18.0-x64.msi"
set "NODE_INSTALLER=node_latest.msi"
powershell -Command "(New-Object Net.WebClient).DownloadFile('%NODE_URL%', '%NODE_INSTALLER%')"
echo Installing Node.js...
msiexec /i "%NODE_INSTALLER%" /qn /norestart
echo Waiting for installation to complete...
timeout /t 10 /nobreak >nul
echo Cleaning up installer...
del "%NODE_INSTALLER%"

:: Refresh PATH from registry
echo Refreshing environment variables...
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path') do set "SysPath=%%b"
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "UserPath=%%b"
set "PATH=%SysPath%;%UserPath%"

:: Check again if Node.js is now available
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js installation completed but node command not found.
    echo Please close this window and run the script again.
    pause
    exit /b 1
)

echo Node.js is now available!
node -v

:run_app
:: Ensure Electron is installed locally
echo Installing Electron if not present...
call npm install --save-dev electron

:: Run the Electron app
echo Starting the app...
call npm start
pause