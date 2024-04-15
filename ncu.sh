ncu -o profile --nvtx --set full -f ./gemm-simple 
ncu -o profile --nvtx --set full -f ./gemm-opt shm
ncu -o profile --nvtx --set full -f ./gemm-opt p1
ncu -o profile --nvtx --set full -f ./gemm-opt p2
ncu -o profile --nvtx --set full -f ./gemm-opt final
