@echo off
setlocal

echo Checking for Node.js...

:: Check if node is installed
where node >nul 2>&1

if %errorlevel%==0 (
    echo Node.js is already installed.
    node -v
    goto end
) else (
    echo Node.js is not installed.
    echo Downloading Node.js installer...

    :: Download the latest LTS x64 Windows installer
    set "NODE_URL=https://nodejs.org/dist/latest-v20.x/node-v20.18.0-x64.msi"
    set "NODE_INSTALLER=node_latest.msi"

    powershell -Command "(New-Object Net.WebClient).DownloadFile('%NODE_URL%', '%NODE_INSTALLER%')"

    echo Running installer...
    msiexec /i "%NODE_INSTALLER%" /qn

    echo Installation completed.
    echo Cleaning up installer file...
    del "%NODE_INSTALLER%"

    echo Node.js should now be installed.
)

:end
echo Done.
pause
