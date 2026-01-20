@echo off
REM Deployment test script for Windows

echo ==========================================
echo   Deployment Verification Test
echo ==========================================
echo.

set API_URL=http://localhost:8880

REM Test 1: Health check
echo 1. Testing health check...
docker-compose exec -T server curl -s http://localhost:8000/health
if errorlevel 0 (
    echo    PASS: Health check
) else (
    echo    FAIL: Health check
)
echo.

REM Test 2: Root path
echo 2. Testing root path...
docker-compose exec -T server curl -s http://localhost:8000/
if errorlevel 0 (
    echo    PASS: Root path
) else (
    echo    FAIL: Root path
)
echo.

REM Test 3: Regions info
echo 3. Testing regions info...
docker-compose exec -T server curl -s http://localhost:8000/regions
if errorlevel 0 (
    echo    PASS: Regions info
) else (
    echo    FAIL: Regions info
)
echo.

REM Test 4: Database connection
echo 4. Testing database connection...
docker-compose exec -T postgres psql -U inference -d inference -c "SELECT 1;" >nul 2>&1
if errorlevel 0 (
    echo    PASS: Database connection
) else (
    echo    FAIL: Database connection
)
echo.

REM Test 5: Redis connection
echo 5. Testing Redis connection...
docker-compose exec -T redis redis-cli ping >nul 2>&1
if errorlevel 0 (
    echo    PASS: Redis connection
) else (
    echo    FAIL: Redis connection
)
echo.

echo ==========================================
echo   All tests completed!
echo ==========================================
echo.
echo Access URLs:
echo    - API Service: %API_URL%
echo    - API Docs: %API_URL%/docs
echo    - Health Check: %API_URL%/health
echo.
pause
