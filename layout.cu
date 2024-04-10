
#include <cuda.h>
#include <stdlib.h>
#include "util.h"

/*
  layout = shape + stride
  逻辑空间 domain
  物理空间 codomain
  shape:   ((最里n层行数，n-1层行数， n-2层行数.... 最外层行数) , (最里n层列数，n-1层列数， n-2层列数.... 最外层列数))

*/

using namespace cute;

int main()
{
    // 1維tensor
    auto shape1 = make_shape(Int<8>{});
    auto stride1 = make_stride(Int<1>{});
    auto layout1 = make_layout(shape1, stride1);
    PRINT("layout1", layout1)

    // 2維tensor
    auto shape2 = make_shape(Int<4>{}, Int<5>{});
    auto stride2 = make_stride(Int<5>{}, Int<1>{});
    auto layout2 = make_layout(shape2, stride2);
    PRINT("layout2", layout2)

    // 3維tensor
    auto shape3 = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    auto stride3 = make_stride(Int<12>{}, Int<4>{}, Int<1>{});
    auto layout3 = make_layout(shape3, stride3);
    PRINT("layout3", layout3)

    // 2維嵌套tensor
    // shape: ((_2,_2),(_2,_4))
    // stride: ((_1,_4),(_2,_8))
    auto shape41 = make_shape(Int<2>{}, Int<2>{});
    auto shape42 = make_shape(Int<2>{}, Int<4>{});
    auto shape4 = make_shape(shape41, shape42);
    auto stride41 = make_stride(Int<1>{}, Int<4>{});
    auto stride42 = make_stride(Int<2>{}, Int<8>{});
    auto stride4 = make_stride(stride41, stride42);
    auto layout4 = make_layout(shape4, stride4);
    PRINT("layout22", layout4)


    // 3維嵌套tensor
    // shape: ((_2,_2,_2),(_2,_4,_4))
    // stride: ((_2,_16,_128),(_1,_4,_32))
    auto shape51 = make_shape(Int<2>{}, Int<2>{}, Int<2>{});
    auto shape52 = make_shape(Int<2>{}, Int<4>{}, Int<4>{});
    auto shape5 = make_shape(shape51, shape52);
    auto stride51 = make_stride(Int<2>{}, Int<16>{}, Int<128>{});
    auto stride52 = make_stride(Int<1>{}, Int<4>{}, Int<32>{});
    auto stride5 = make_stride(stride51, stride52);
    auto layout5 = make_layout(shape5, stride5);
    PRINT("layout33", layout5)

}