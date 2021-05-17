#include "hip/hip_runtime.h"
/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "../mc_rocm.h"
#ifdef __cplusplus
}
#endif

__global__ void wait_kernel(volatile uint32_t *status) {
    ucc_status_t st;
    *status = UCC_MC_ROCM_TASK_STARTED;
    do {
        st = (ucc_status_t)*status;
    } while(st != UCC_MC_ROCM_TASK_COMPLETED);
    *status = UCC_MC_ROCM_TASK_COMPLETED_ACK;
    return;
}

__global__ void wait_kernel_nb(volatile uint32_t *status) {
    *status = UCC_MC_ROCM_TASK_COMPLETED_ACK;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_mc_rocm_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 hipStream_t stream)
{
    if (blocking_wait) {
        hipLaunchKernelGGL(wait_kernel, dim3(1), dim3(1), 0, stream, status);
    } else {
        hipLaunchKernelGGL(wait_kernel_nb, dim3(1), dim3(1), 0, stream, status);
    }
    ROCMCHECK(hipGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
