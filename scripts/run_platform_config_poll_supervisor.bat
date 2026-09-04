@echo off
setlocal
if "%~1"=="" exit /b 2
set "B4_BACKEND=%~1"
set "B4_CREDENTIAL_ENV=%~2"
if "%B4_CREDENTIAL_ENV%"=="" set "B4_CREDENTIAL_ENV=TIANYAN_LOGIN_KEY"
:restart
python "%~dp0poll_platform_config.py" --backend "%B4_BACKEND%" --credential-env "%B4_CREDENTIAL_ENV%"
timeout /t 60 /nobreak >nul
goto restart
