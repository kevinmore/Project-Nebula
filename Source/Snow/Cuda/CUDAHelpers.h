#pragma once
#include <Utility/CUDAInclude.h>

#define LAUNCH( ... ) { __VA_ARGS__; checkCudaErrors( cudaDeviceSynchronize() ); }

#define cudaMallocAndCopy( dst, src, size )                    \
({                                                             \
    cudaMalloc((void**) &dst, size);                           \
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);        \
})

#define TEST( toCheck, msg, failExprs)      \
({                                          \
    if (toCheck){                           \
        printf("[PASSED]: %s\n", msg);      \
    }else{                                  \
        printf("[FAILED]: %s\n", msg);      \
        failExprs;                          \
    }                                       \
})

#define THREAD_COUNT 128
