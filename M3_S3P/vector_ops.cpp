#include <cstdio>
#include <cstdlib>
#include <CL/cl.h>
#include <chrono>

#define PRINT 1

int SZ = 100000000;

int *v1, *v2, *v_out;
cl_mem bufV1, bufV2, bufV_out;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_kernel kernel;
cl_command_queue queue;
cl_event event = NULL;
int err;

cl_device_id create_device();
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
void setup_kernel_memory();
void copy_kernel_args();
void free_memory();
void init(int *&A, int size);
void print(int *A, int size);

int main(int argc, char **argv)
{
    if (argc > 1)
    {
        SZ = atoi(argv[1]);
    }

    // Host arrays are allocated first so the input data can be prepared before device transfer.
    init(v1, SZ);
    init(v2, SZ);
    init(v_out, SZ);

    size_t global[1] = {(size_t)SZ};

    print(v1, SZ);
    print(v2, SZ);

    // OpenCL is used here because vector addition is data-parallel and maps cleanly to many work-items.
    setup_openCL_device_context_queue_kernel((char *)"./vector_ops_ocl.cl", (char *)"vector_add_ocl");

    // Device buffers are created once, then the input vectors are copied across in a single transfer.
    setup_kernel_memory();

    // The kernel receives the problem size and the three buffers it needs to compute the result.
    copy_kernel_args();

    // Timing begins at kernel launch so the measured phase focuses on accelerator execution and readback.
    auto start = std::chrono::high_resolution_clock::now();
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);
    clEnqueueReadBuffer(queue, bufV_out, CL_TRUE, 0, SZ * sizeof(int), &v_out[0], 0, NULL, NULL);
    auto stop = std::chrono::high_resolution_clock::now();

    print(v_out, SZ);

    std::chrono::duration<double, std::milli> elapsed_time = stop - start;
    printf("Kernel Execution Time: %f ms\n", elapsed_time.count());

    free_memory();
    return 0;
}

void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size);
    if (A == NULL)
    {
        perror("malloc failed");
        exit(1);
    }

    for (long i = 0; i < size; i++)
    {
        A[i] = rand() % 100;
    }
}

void print(int *A, int size)
{
    if (PRINT == 0)
    {
        return;
    }

    if (PRINT == 1 && size > 15)
    {
        for (long i = 0; i < 5; i++)
        {
            printf("%d ", A[i]);
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++)
        {
            printf("%d ", A[i]);
        }
    }
    else
    {
        for (long i = 0; i < size; i++)
        {
            printf("%d ", A[i]);
        }
    }
    printf("\n----------------------------\n");
}

void free_memory()
{
    // Release OpenCL objects first, then free host memory to avoid leaks.
    clReleaseMemObject(bufV1);
    clReleaseMemObject(bufV2);
    clReleaseMemObject(bufV_out);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    free(v1);
    free(v2);
    free(v_out);
}

void copy_kernel_args()
{
    err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(int), (void *)&SZ);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufV1);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufV2);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&bufV_out);

    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d\n", err);
        exit(1);
    }
}

void setup_kernel_memory()
{
    // Read-write buffers are used because the host writes inputs and later reads back the output.
    bufV1 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV2 = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);
    bufV_out = clCreateBuffer(context, CL_MEM_READ_WRITE, SZ * sizeof(int), NULL, NULL);

    if (bufV1 == NULL || bufV2 == NULL || bufV_out == NULL)
    {
        perror("Couldn't create OpenCL buffers");
        exit(1);
    }

    clEnqueueWriteBuffer(queue, bufV1, CL_TRUE, 0, SZ * sizeof(int), &v1[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufV2, CL_TRUE, 0, SZ * sizeof(int), &v2[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int status;

    // A single-device context and queue are enough because this task is a straightforward vector-add benchmark.
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
    if (status < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    // Use the compatible command-queue API so the code runs on both older and newer OpenCL installations.
    queue = clCreateCommandQueue(context, device_id, 0, &status);
    if (status < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    }

    kernel = clCreateKernel(program, kernelname, &status);
    if (status < 0)
    {
        perror("Couldn't create a kernel");
        printf("error = %d\n", status);
        exit(1);
    }
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    if (program_buffer == NULL)
    {
        perror("malloc failed");
        exit(1);
    }
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // The kernel source is compiled at runtime so build errors can be captured immediately.
    program = clCreateProgramWithSource(ctx, 1, (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

cl_device_id create_device()
{
    cl_platform_id platform;
    cl_device_id dev;
    int status;

    status = clGetPlatformIDs(1, &platform, NULL);
    if (status < 0)
    {
        perror("Couldn't identify a platform");
        exit(1);
    }

    // Prefer GPU execution for parallel throughput, then fall back to CPU if no GPU is available.
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (status == CL_DEVICE_NOT_FOUND)
    {
        printf("GPU not found\n");
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if (status < 0)
    {
        perror("Couldn't access any devices");
        exit(1);
    }

    return dev;
}