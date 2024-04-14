nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o gemm-simple ./gemm-simple
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o gemm-opt-shm ./gemm-opt shm
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o gemm-opt-p1 ./gemm-opt p1
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o gemm-opt-p2 ./gemm-opt p2
nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx -o gemm-opt-final ./gemm-opt final
