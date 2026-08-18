@echo off
setlocal

set "REPO_ROOT=%~dp0.."
cd /d "%REPO_ROOT%"

if not defined TEST_PATH set "TEST_PATH=tests\python"
if not defined PYTEST_BASETEMP set "PYTEST_BASETEMP=%TEMP%\marketmind-pytest-%RANDOM%"
if not defined PYTEST_BIN set "PYTEST_BIN=pytest"

if "%~1"=="-h" goto :help
if "%~1"=="--help" goto :help

echo Clearing marketmind.log
if not exist logs mkdir logs
if not exist tests\logs mkdir tests\logs
echo. > logs\marketmind.log
echo. > tests\logs\marketmind.log

echo Repo root: %CD%
echo Test path: %TEST_PATH%
echo Pytest basetemp: %PYTEST_BASETEMP%

"%PYTEST_BIN%" -v --basetemp="%PYTEST_BASETEMP%" %TEST_PATH% %*
exit /b %ERRORLEVEL%

:help
echo Usage: tests\run_tests.bat [pytest selectors/flags...]
echo.
echo Environment variables:
echo   TEST_PATH         target path ^(default: tests\python^)
echo   PYTEST_BASETEMP   pytest temporary root
echo   PYTEST_BIN        pytest binary ^(default: pytest^)
echo.
echo Examples:
echo   tests\run_tests.bat
echo   set TEST_PATH=tests\python\unit ^&^& tests\run_tests.bat --no-cov
exit /b 0
