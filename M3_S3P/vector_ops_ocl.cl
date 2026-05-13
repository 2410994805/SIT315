__kernel void vector_add_ocl(const int size,
                             __global int *v1,
                             __global int *v2,
                             __global int *v_out)
{
    // Each work-item processes exactly one element, which is ideal for vector addition.
    const int globalIndex = get_global_id(0);

    if (globalIndex < size)
    {
        v_out[globalIndex] = v1[globalIndex] + v2[globalIndex];
    }
}