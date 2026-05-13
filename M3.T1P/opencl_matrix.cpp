// Libraries
#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <cstring>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;
using namespace std::chrono;

// Fill matrix with random values 0-99
void fillMatrixWithRandomValues(vector<int> &matrix, int matrixSize)
{
    for (int i = 0; i < matrixSize * matrixSize; i++)
    {
        matrix[i] = rand() % 100;
    }
}

// Calculate checksum for correctness verification
long long calculateChecksum(const vector<int> &matrix)
{
    long long totalChecksum = 0;
    for (int value : matrix)
    {
        totalChecksum += value;
    }
    return totalChecksum;
}

// Write matrix to output file in row-major format
void writeMatrixToFile(const vector<int> &matrix, int matrixSize, const string &fileName)
{
    ofstream outputFile(fileName);
    if (!outputFile)
    {
        return;
    }

    for (int i = 0; i < matrixSize; i++)
    {
        for (int j = 0; j < matrixSize; j++)
        {
            outputFile << matrix[i * matrixSize + j] << " ";
        }
        outputFile << "\n";
    }

    outputFile.close();
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        cout << "Usage: ./opencl_matrix N [write_output]\n";
        return 1;
    }

    int matrixSize = atoi(argv[1]);
    int writeOutput = (argc >= 3) ? atoi(argv[2]) : 0;

    if (matrixSize <= 0)
    {
        cout << "N must be > 0\n";
        return 1;
    }

    vector<int> matrixA(matrixSize * matrixSize);
    vector<int> matrixB(matrixSize * matrixSize);
    vector<int> matrixC(matrixSize * matrixSize, 0);

    srand(0);
    fillMatrixWithRandomValues(matrixA, matrixSize);
    fillMatrixWithRandomValues(matrixB, matrixSize);

    const char *kernelSource = R"CLC(
        __kernel void matrixMultiply(const int N,
                                     __global const int* A,
                                     __global const int* B,
                                     __global int* C)
        {
            int row = get_global_id(0);
            int col = get_global_id(1);

            if (row < N && col < N)
            {
                int cellSum = 0;
                for (int k = 0; k < N; k++)
                {
                    cellSum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = cellSum;
            }
        }
    )CLC";

    cl_int errorCode;

    cl_uint platformCount = 0;
    errorCode = clGetPlatformIDs(0, nullptr, &platformCount);
    if (errorCode != CL_SUCCESS || platformCount == 0)
    {
        cout << "No OpenCL platform found\n";
        return 1;
    }

    vector<cl_platform_id> platforms(platformCount);
    clGetPlatformIDs(platformCount, platforms.data(), nullptr);

    cl_device_id selectedDevice = nullptr;
    cl_platform_id selectedPlatform = nullptr;
    cl_device_type selectedType = 0;

    // Prefer GPU, then CPU
    for (cl_platform_id platform : platforms)
    {
        cl_uint deviceCount = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount) == CL_SUCCESS && deviceCount > 0)
        {
            vector<cl_device_id> devices(deviceCount);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr);
            selectedDevice = devices[0];
            selectedPlatform = platform;
            selectedType = CL_DEVICE_TYPE_GPU;
            break;
        }
    }

    if (selectedDevice == nullptr)
    {
        for (cl_platform_id platform : platforms)
        {
            cl_uint deviceCount = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount) == CL_SUCCESS && deviceCount > 0)
            {
                vector<cl_device_id> devices(deviceCount);
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, deviceCount, devices.data(), nullptr);
                selectedDevice = devices[0];
                selectedPlatform = platform;
                selectedType = CL_DEVICE_TYPE_CPU;
                break;
            }
        }
    }

    if (selectedDevice == nullptr)
    {
        cout << "No suitable OpenCL device found\n";
        return 1;
    }

    char deviceName[256] = {0};
    clGetDeviceInfo(selectedDevice, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &selectedDevice, nullptr, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        cout << "Failed to create OpenCL context\n";
        return 1;
    }

    cl_command_queue commandQueue = clCreateCommandQueue(context, selectedDevice, 0, &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        cout << "Failed to create command queue\n";
        clReleaseContext(context);
        return 1;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        cout << "Failed to create OpenCL program\n";
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        return 1;
    }

    errorCode = clBuildProgram(program, 1, &selectedDevice, nullptr, nullptr, nullptr);
    if (errorCode != CL_SUCCESS)
    {
        size_t logSize = 0;
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, selectedDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);

        cout << "OpenCL build failed\n";
        cout << buildLog.data() << '\n';

        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrixMultiply", &errorCode);
    if (errorCode != CL_SUCCESS)
    {
        cout << "Failed to create kernel\n";
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        return 1;
    }

    size_t bytes = matrixSize * matrixSize * sizeof(int);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &errorCode);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &errorCode);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &errorCode);

    if (errorCode != CL_SUCCESS)
    {
        cout << "Failed to create OpenCL buffers\n";
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        return 1;
    }

    auto start = high_resolution_clock::now();

    clEnqueueWriteBuffer(commandQueue, bufferA, CL_TRUE, 0, bytes, matrixA.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(commandQueue, bufferB, CL_TRUE, 0, bytes, matrixB.data(), 0, nullptr, nullptr);

    clSetKernelArg(kernel, 0, sizeof(int), &matrixSize);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferC);

    size_t globalWorkSize[2] = {(size_t)matrixSize, (size_t)matrixSize};

    clEnqueueNDRangeKernel(commandQueue, kernel, 2, nullptr, globalWorkSize, nullptr, 0, nullptr, nullptr);
    clFinish(commandQueue);

    clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, bytes, matrixC.data(), 0, nullptr, nullptr);

    auto end = high_resolution_clock::now();
    long long executionTimeMicroseconds = duration_cast<microseconds>(end - start).count();

    long long checksum = calculateChecksum(matrixC);

    if (writeOutput == 1)
    {
        writeMatrixToFile(matrixC, matrixSize, "C_opencl.txt");
    }

    cout << "N = " << matrixSize << '\n';
    cout << "Processes = 1\n";
    cout << "Threads per process = device managed\n";
    cout << "Method = OpenCL\n";
    cout << "Device = " << deviceName << '\n';
    cout << "Device type = " << (selectedType == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") << '\n';
    cout << "Execution time = " << executionTimeMicroseconds
         << " us (" << executionTimeMicroseconds / 1000.0 << " ms)\n";
    cout << "Checksum = " << checksum << '\n';
    cout << "Output = " << (writeOutput == 1 ? "C_opencl.txt" : "not written") << '\n';

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return 0;
}