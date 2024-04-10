#include <cuda.h>
#include <stdlib.h>
#include <cute/tensor.hpp>

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_v0(half *z, half *x, half *y, int num, const half a, const half b, const half c)
{
    using namespace cute;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = tid * kNumElemPerThread;
    if (tid > num / kNumElemPerThread)
    {
        return;
    }
    half2 a2 = {a, a};
    half2 b2 = {b, b};
    half2 c2 = {c, c};
    half2 *x_ptr = reinterpret_cast<half2 *>(x + offset);
    half2 *y_ptr = reinterpret_cast<half2 *>(y + offset);
    half2 *z_ptr = reinterpret_cast<half2 *>(z + offset);
#pragma unroll
    for (int i = 0; i < kNumElemPerThread/2; i++)
    {
        half2 x_val, y_val, result;
        x_val = __ldg(x_ptr++);
        y_val = __ldg(y_ptr++);
        result = a2 * x_val + b2 * y_val + c2;
        *(z_ptr++) = result;
    }
}

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_v1(half *z, half *x, half *y, int num, const half a, const half b, const half c)
{
    using namespace cute;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid > num / kNumElemPerThread)
    {
        return;
    }

    auto tZ = make_tensor(make_gmem_ptr(z), make_shape(num));
    auto tX = make_tensor(make_gmem_ptr(x), make_shape(num));
    auto tY = make_tensor(make_gmem_ptr(y), make_shape(num));

    auto tZg = local_tile(tZ, make_shape(Int<kNumElemPerThread>()), tid);
    auto tXg = local_tile(tX, make_shape(Int<kNumElemPerThread>()), tid);
    auto tYg = local_tile(tY, make_shape(Int<kNumElemPerThread>()), tid);

    auto tZr = make_tensor_like(tZg);
    auto tXr = make_tensor_like(tXg);
    auto tYr = make_tensor_like(tYg);

    // LDG.128
    cute::copy(tXg, tXr);
    cute::copy(tYg, tYr);

#pragma unroll
    for (int i = 0; i < size(tXr); i++)
    {
        tZr(i) = a * tXr(i) + (b * tYr(i) + c);
    }
    // STG.128
    cute::copy(tZr, tZg);
}

// z = ax + by + c
template <int kNumElemPerThread = 8>
__global__ void vector_add_v2(
    half *z, const half *x, const half *y, int num, const half a, const half b, const half c)
{
    using namespace cute;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num / kNumElemPerThread)
    {
        return;
    }

    Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
    Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
    Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

    Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
    Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
    Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

    Tensor txR = make_tensor_like(txr);
    Tensor tyR = make_tensor_like(tyr);
    Tensor tzR = make_tensor_like(tzr);

    // LDG.128
    copy(txr, txR);
    copy(tyr, tyR);

    half2 a2 = {a, a};
    half2 b2 = {b, b};
    half2 c2 = {c, c};

    auto tzR2 = recast<half2>(tzR);
    auto txR2 = recast<half2>(txR);
    auto tyR2 = recast<half2>(tyR);

#pragma unroll
    for (int i = 0; i < size(tzR2); ++i)
    {
        // two hfma2 instruction
        tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
    }

    auto tzRx = recast<half>(tzR2);
    // STG.128
    copy(tzRx, tzr);
}

int main()
{
    const int numElementPerThread = 8;
    const half a = 2.0;
    const half b = 1.0;
    const half c = 1.0;
    unsigned int size = 1024 * 8192;

    cudaEvent_t start, end;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    half *cx = (half *)malloc(size * sizeof(half));
    half *cy = (half *)malloc(size * sizeof(half));
    half *cz = (half *)malloc(size * sizeof(half));
    for (int i = 0; i < size; i++)
    {
        cx[i] = 1;
        cy[i] = 1;
        cz[i] = 0;
    }
    half *gx;
    half *gy;
    half *gz;
    cudaMalloc(&gx, size * sizeof(half));
    cudaMalloc(&gy, size * sizeof(half));
    cudaMalloc(&gz, size * sizeof(half));
    cudaMemcpy(gx, cx, size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gy, cy, size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(gz, cz, size * sizeof(half), cudaMemcpyHostToDevice);

    int block_size = 1024;
    int grid = size / (block_size * numElementPerThread);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        vector_add_v0<numElementPerThread><<<grid, block_size>>>(gz, gx, gy, size, a, b, c);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "vector_add_v0 took " << elapsedTime / 100 << "ms." << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        vector_add_v1<numElementPerThread><<<grid, block_size>>>(gz, gx, gy, size, a, b, c);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "vector_add_v1 took " << elapsedTime / 100 << "ms." << std::endl;

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
    {
        vector_add_v2<numElementPerThread><<<grid, block_size>>>(gz, gx, gy, size, a, b, c);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "vector_add_v2 took " << elapsedTime / 100 << "ms." << std::endl;
}