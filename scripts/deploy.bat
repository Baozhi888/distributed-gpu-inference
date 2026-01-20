@echo off
REM Deploy script for Windows

echo ==========================================
echo   Distributed GPU Inference Platform
echo ==========================================
echo.

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker not installed
    echo Please install Docker Desktop first
    pause
    exit /b 1
)

REM Check Docker Compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Docker Compose not installed
        pause
        exit /b 1
    )
)

REM Check .env file
if not exist .env (
    echo WARNING: .env file not found, using defaults
)

echo Environment check passed
echo.

REM Select deployment mode
echo Select deployment mode:
echo 1) Server only (Server + Database + Redis)
echo 2) Server + Worker (requires NVIDIA GPU)
echo.
set /p choice="Enter option [1-2]: "

if "%choice%"=="1" (
    echo.
    echo Starting server mode...
    docker-compose up -d postgres redis server
) else if "%choice%"=="2" (
    echo.
    echo Starting server + worker mode...
    docker-compose --profile with-worker up -d
) else (
    echo Invalid option
    pause
    exit /b 1
)

echo.
echo Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service status
echo.
echo Service status:
docker-compose ps

echo.
echo ==========================================
echo   Deployment Complete!
echo ==========================================
echo.
echo Access URLs:
echo    - API Service: http://localhost:8880
echo    - Health Check: http://localhost:8880/health
echo    - API Docs: http://localhost:8880/docs
echo.
echo View logs:
echo    docker-compose logs -f server
echo.
echo Stop services:
echo    docker-compose down
echo.
pause
