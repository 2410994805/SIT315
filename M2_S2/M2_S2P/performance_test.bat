@echo off
REM SIT315 M2.S2P - Automated Performance Test Script
REM Author  : Jahan Garg (2410994805)
REM Purpose : Compiles all three programs and benchmarks pthread and
REM           std::thread across 1, 2, 4, 8 threads on Windows (MinGW).
REM Requires: MinGW-w64 g++ available in system PATH

echo SIT315 M2.S2P - Performance Testing
echo Author: Jahan Garg (2410994805)
echo.

echo [1/3] Compiling sequential baseline...
g++ -std=c++11 -O2 sequential.cpp -o sequential.exe
if %errorlevel% neq 0 ( echo COMPILE ERROR: sequential.cpp & pause & exit /b 1 )

echo [2/3] Compiling pthread implementation...
g++ -std=c++11 -O2 -pthread parallel_pthread.cpp -o parallel_pthread.exe
if %errorlevel% neq 0 ( echo COMPILE ERROR: parallel_pthread.cpp & pause & exit /b 1 )

echo [3/3] Compiling std::thread implementation...
g++ -std=c++11 -O2 parallel_stdthread.cpp -o parallel_stdthread.exe
if %errorlevel% neq 0 ( echo COMPILE ERROR: parallel_stdthread.cpp & pause & exit /b 1 )

echo All compilations successful.
echo.

echo SEQUENTIAL BASELINE
sequential.exe
echo.

echo PTHREAD IMPLEMENTATION
for %%t in (1 2 4 8) do (
    echo Testing with %%t thread(s):
    parallel_pthread.exe %%t
    echo.
)

echo STD::THREAD IMPLEMENTATION
for %%t in (1 2 4 8) do (
    echo Testing with %%t thread(s):
    parallel_stdthread.exe %%t
    echo.
)

echo Testing complete. Record results in your performance report.
pause
