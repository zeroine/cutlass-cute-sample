#include <cuda.h>
#include <stdlib.h>
#include "util.h"

/*
    cute中的Tensor更多的是对Tensor进行分解和组合等操作，而这些操作多是对Layout的变换（只是逻辑层面的数据组织形式），底层的数据实体一般不变更。
    Tensor = Layout + storage
*/
using namespace cute;
using namespace std;

__global__ void handle_regiser_tensor()
{
    auto rshape = make_shape(Int<4>{}, Int<2>{});
    auto rstride = make_stride(Int<2>{}, Int<1>{});
    auto rtensor = make_tensor(make_layout(rshape, rstride));

    PRINT("rtensor.layout", rtensor.layout());
    PRINT("rtensor.shape", rtensor.shape());
    PRINT("rtensor.stride", rtensor.stride());
    PRINT("rtensor.size", rtensor.size());
    PRINT("rtensor.data", rtensor.data());

    print("\n");
}

__global__ void handle_global_tensor(int *pointer)
{
    auto gshape = make_shape(Int<4>{}, Int<6>{});
    auto gstride = make_stride(Int<6>{}, Int<1>{});
    // need in device function, not host function
    auto gtensor = make_tensor(make_gmem_ptr(pointer), make_layout(gshape, gstride));
    PRINTTENSOR("global tensor", gtensor);

    auto coord = make_coord(2, 1);
    PRINT("gtensor(2,1)", gtensor(coord));

    auto tensor_slice = gtensor(_, 1);
    PRINTTENSOR("tensor slice", tensor_slice);

    auto tensor_tile = local_tile(gtensor, make_shape(Int<2>(), Int<2>()), make_coord(Int<1>(), Int<1>()));
    PRINTTENSOR("tensor tile (2,2) index (1,1)", tensor_tile);

    int thr_idx = 1;
    auto tensor_partition = local_partition(gtensor, Layout<Shape<_2, _2>, Stride<_2, _1>>{}, thr_idx);
    PRINTTENSOR("tensor partition tile (2,2) index (1)", tensor_partition);
}

int main()
{
    // register tensor
    handle_regiser_tensor<<<1, 1>>>();

    // global memory tensor
    int *pointer;
    int size = 4 * 6;
    cudaMalloc(&pointer, size * sizeof(int));
    int *cpointer = (int *)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        cpointer[i] = i;
    }
    cudaMemcpy(pointer, cpointer, size * sizeof(int), cudaMemcpyHostToDevice);
    handle_global_tensor<<<1, 1>>>(pointer);
    cudaDeviceSynchronize();

    // copy tensor
    auto rshape = make_shape(Int<4>{}, Int<2>{});
    auto rstride = make_stride(Int<2>{}, Int<1>{});
    auto rtensor = make_tensor(make_layout(rshape, rstride));
    auto ctensor = make_fragment_like(rtensor);
    PRINT("ctensor.layout", ctensor.layout());
}