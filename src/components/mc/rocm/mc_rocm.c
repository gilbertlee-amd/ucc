/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_rocm.h"
#include "utils/ucc_malloc.h"
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

static const char *stream_task_modes[] = {
    [UCC_MC_ROCM_TASK_KERNEL]  = "kernel",
    [UCC_MC_ROCM_TASK_MEM_OPS] = "driver",
    [UCC_MC_ROCM_TASK_AUTO]    = "auto",
    [UCC_MC_ROCM_TASK_LAST]    = NULL
};

static const char *task_stream_types[] = {
    [UCC_MC_ROCM_USER_STREAM]      = "user",
    [UCC_MC_ROCM_INTERNAL_STREAM]  = "ucc",
    [UCC_MC_ROCM_TASK_STREAM_LAST] = NULL
};

static ucc_config_field_t ucc_mc_rocm_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_rocm_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction",
     ucc_offsetof(ucc_mc_rocm_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

    {"STREAM_TASK_MODE", "auto",
     "Mechanism to create stream dependency\n"
     "kernel - use waiting kernel\n"
     "driver - use driver MEM_OPS\n"
     "auto   - runtime automatically chooses best one",
     ucc_offsetof(ucc_mc_rocm_config_t, strm_task_mode),
     UCC_CONFIG_TYPE_ENUM(stream_task_modes)},

    {"TASK_STREAM", "user",
     "Stream for rocm task\n"
     "user - user stream provided in execution engine context\n"
     "ucc  - ucc library internal stream",
     ucc_offsetof(ucc_mc_rocm_config_t, task_strm_type),
     UCC_CONFIG_TYPE_ENUM(task_stream_types)},

    {"STREAM_BLOCKING_WAIT", "1",
     "Stream is blocked until collective operation is done",
     ucc_offsetof(ucc_mc_rocm_config_t, stream_blocking_wait),
     UCC_CONFIG_TYPE_UINT},

    {NULL}
};

static ucc_status_t ucc_mc_rocm_stream_req_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                              size_t *size_p,
                                                              void ** chunk_p)
{
    ucc_status_t status;

    status = ROCM_FUNC(hipHostMalloc((void**)chunk_p, *size_p,
                       hipHostMallocMapped));
    return status;
}

static void ucc_mc_rocm_stream_req_mpool_chunk_free(ucc_mpool_t *mp,
                                                    void *       chunk)
{
    hipHostFree(chunk);
}

static void ucc_mc_rocm_stream_req_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_rocm_stream_request_t *req = (ucc_mc_rocm_stream_request_t*) obj;

    ROCM_FUNC(hipHostGetDevicePointer(
                  (void**)(&req->dev_status), (void *)&req->status, 0));
}

static ucc_mpool_ops_t ucc_mc_rocm_stream_req_mpool_ops = {
    .chunk_alloc   = ucc_mc_rocm_stream_req_mpool_chunk_malloc,
    .chunk_release = ucc_mc_rocm_stream_req_mpool_chunk_free,
    .obj_init      = ucc_mc_rocm_stream_req_init,
    .obj_cleanup   = NULL
};

static void ucc_mc_rocm_event_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_rocm_event_t *base = (ucc_mc_rocm_event_t *) obj;

    if (hipSuccess != hipEventCreateWithFlags(&base->event,
                                                hipEventDisableTiming)) {
        mc_error(&ucc_mc_rocm.super, "hipEventCreateWithFlags Failed");
    }
}

static void ucc_mc_rocm_event_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_mc_rocm_event_t *base = (ucc_mc_rocm_event_t *) obj;
    if (hipSuccess != hipEventDestroy(base->event)) {
        mc_error(&ucc_mc_rocm.super, "hipEventDestroy Failed");
    }
}

static ucc_mpool_ops_t ucc_mc_rocm_event_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_mc_rocm_event_init,
    .obj_cleanup   = ucc_mc_rocm_event_cleanup,
};

static ucc_status_t ucc_mc_rocm_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 hipStream_t stream)
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_post_driver_stream_task(uint32_t *status,
                                                        int blocking_wait,
                                                        hipStream_t stream)
{
#if 0
    hipDeviceptr_t status_ptr  = (hipDeviceptr_t)status;

    if (blocking_wait) {
        ROCMDRV_FUNC(cuStreamWriteValue32(stream, status_ptr,
                                          UCC_MC_ROCM_TASK_STARTED, 0));
        ROCMDRV_FUNC(cuStreamWaitValue32(stream, status_ptr,
                                         UCC_MC_ROCM_TASK_COMPLETED,
                                         hip_STREAM_WAIT_VALUE_EQ));
    }
    ROCMDRV_FUNC(cuStreamWriteValue32(stream, status_ptr,
                                      UCC_MC_ROCM_TASK_COMPLETED_ACK, 0));
#endif
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_init()
{
    ucc_mc_rocm_config_t *cfg = MC_ROCM_CONFIG;
    struct hipDeviceProp_t prop;
    ucc_status_t status;
    int device, num_devices;
    int mem_ops_attr = 1;
    //hipDevice_t hip_dev;
    hipError_t rocm_st;

    ucc_strncpy_safe(ucc_mc_rocm.super.config->log_component.name,
                     ucc_mc_rocm.super.super.name,
                     sizeof(ucc_mc_rocm.super.config->log_component.name));
    rocm_st = hipGetDeviceCount(&num_devices);
    if ((rocm_st != hipSuccess) || (num_devices == 0)) {
        mc_info(&ucc_mc_rocm.super, "rocm devices are not found");
        return UCC_ERR_NO_RESOURCE;
    }
    ROCMCHECK(hipGetDevice(&device));
    ROCMCHECK(hipGetDeviceProperties(&prop, device));
    cfg->reduce_num_threads = prop.maxThreadsPerBlock;
    if (cfg->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        if (prop.maxGridSize[0] < cfg->reduce_num_blocks) {
            mc_warn(&ucc_mc_rocm.super, "number of blocks is too large, "
                    "max supported %d", prop.maxGridSize[0]);
            cfg->reduce_num_blocks = prop.maxGridSize[0];
        }
    }

    ROCMCHECK(hipStreamCreateWithFlags(&ucc_mc_rocm.stream,
              hipStreamNonBlocking));

    /*create event pool */
    status = ucc_mpool_init(&ucc_mc_rocm.events, 0, sizeof(ucc_mc_rocm_event_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_mc_rocm_event_mpool_ops, UCC_THREAD_MULTIPLE,
                            "ROCM Event Objects");
    if (status != UCC_OK) {
        mc_error(&ucc_mc_rocm.super, "Error to create event pool");
        return status;
    }

    /* create request pool */
    status = ucc_mpool_init(
        &ucc_mc_rocm.strm_reqs, 0, sizeof(ucc_mc_rocm_stream_request_t), 0,
        UCC_CACHE_LINE_SIZE, 16, UINT_MAX, &ucc_mc_rocm_stream_req_mpool_ops,
        UCC_THREAD_MULTIPLE, "ROCM Event Objects");
    if (status != UCC_OK) {
        mc_error(&ucc_mc_rocm.super, "Error to create event pool");
        return status;
    }

    if (cfg->strm_task_mode == UCC_MC_ROCM_TASK_KERNEL) {
        ucc_mc_rocm.strm_task_mode = UCC_MC_ROCM_TASK_KERNEL;
        ucc_mc_rocm.post_strm_task = ucc_mc_rocm_post_kernel_stream_task;
    } else {
        ucc_mc_rocm.strm_task_mode = UCC_MC_ROCM_TASK_MEM_OPS;
        ucc_mc_rocm.post_strm_task = ucc_mc_rocm_post_driver_stream_task;
#if 0
        ROCMDRV_FUNC(hipCtxGetDevice(&hip_dev));
        ROCMDRV_FUNC(hipDeviceGetAttribute(&mem_ops_attr,
                    HIP_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                    hip_dev));
#endif

        if (cfg->strm_task_mode == UCC_MC_ROCM_TASK_AUTO) {
            if (mem_ops_attr == 0) {
                mc_info(&ucc_mc_rocm.super,
                        "ROCM MEM OPS are not supported or disabled");
                ucc_mc_rocm.strm_task_mode = UCC_MC_ROCM_TASK_KERNEL;
                ucc_mc_rocm.post_strm_task = ucc_mc_rocm_post_kernel_stream_task;
            }
        } else if (mem_ops_attr == 0) {
            mc_error(&ucc_mc_rocm.super,
                     "ROCM MEM OPS are not supported or disabled");
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    ucc_mc_rocm.task_strm_type = cfg->task_strm_type;

    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_finalize()
{
    ROCMCHECK(hipStreamDestroy(ucc_mc_rocm.stream));
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_alloc(void **ptr, size_t size)
{
    hipError_t st;

    st = hipMalloc(ptr, size);
    if (st != hipSuccess) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to allocate %zd bytes, "
                 "rocm error %d(%s)",
                 size, st, hipGetErrorString(st));
        return UCC_ERR_NO_MEMORY;
    }

    mc_debug(&ucc_mc_rocm.super, "ucc_mc_rocm_mem_alloc size:%ld", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_free(void *ptr)
{
    hipError_t st;

    st = hipFree(ptr);
    if (st != hipSuccess) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to free mem at %p, "
                 "rocm error %d(%s)",
                 ptr, st, hipGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_memcpy(void *dst, const void *src, size_t len,
                                       ucc_memory_type_t dst_mem,
                                       ucc_memory_type_t src_mem)
{
    hipError_t    st;
    ucc_assert(dst_mem == UCC_MEMORY_TYPE_ROCM ||
               src_mem == UCC_MEMORY_TYPE_ROCM);

    st = hipMemcpyAsync(dst, src, len, hipMemcpyDefault, ucc_mc_rocm.stream);
    if (st != hipSuccess) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to launch hipMemcpyAsync,  dst %p, src %p, len %zd "
                 "rocm error %d(%s)",
                 dst, src, len, st, hipGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    st = hipStreamSynchronize(ucc_mc_rocm.stream);
    if (st != hipSuccess) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to synchronize mc_rocm.stream "
                 "rocm error %d(%s)",
                 st, hipGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_query(const void *ptr,
                                          size_t length,
                                          ucc_mem_attr_t *mem_attr)
{
    struct hipPointerAttribute_t attr;
    hipError_t                  st;
    hipError_t                     hip_err;
    ucc_memory_type_t            mem_type;
    void                         *base_address;
    size_t                       alloc_length;

    if (!(mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_MEM_TYPE     |
                                  UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                                  UCC_MEM_ATTR_FIELD_ALLOC_LENGTH))) {
        return UCC_OK;
    }

    if (ptr == 0) {
        mem_type = UCC_MEMORY_TYPE_HOST;
    } else {
        if (mem_attr->field_mask & UCC_MEM_ATTR_FIELD_MEM_TYPE) {
            st = hipPointerGetAttributes(&attr, ptr);
            if (st != hipSuccess) {
                hipGetLastError();
                return UCC_ERR_NOT_SUPPORTED;
            }
#if ROCMRT_VERSION >= 10000
            switch (attr.type) {
            case rocmMemoryTypeHost:
                mem_type = UCC_MEMORY_TYPE_HOST;
                break;
            case rocmMemoryTypeDevice:
                mem_type = UCC_MEMORY_TYPE_ROCM;
                break;
            case rocmMemoryTypeManaged:
                mem_type = UCC_MEMORY_TYPE_ROCM_MANAGED;
                break;
            default:
                return UCC_ERR_NOT_SUPPORTED;
            }
#else
            if (attr.memoryType == hipMemoryTypeDevice) {
                if (attr.isManaged) {
                    mem_type = UCC_MEMORY_TYPE_ROCM_MANAGED;
                } else {
                    mem_type = UCC_MEMORY_TYPE_ROCM;
                }
            }
            else if (attr.memoryType == hipMemoryTypeHost) {
                mem_type = UCC_MEMORY_TYPE_HOST;
            } else {
                return UCC_ERR_NOT_SUPPORTED;
            }
#endif
            mem_attr->mem_type = mem_type;
        }

        if (mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_ALLOC_LENGTH |
                                    UCC_MEM_ATTR_FIELD_BASE_ADDRESS)) {
            hip_err = hipMemGetAddressRange((hipDeviceptr_t*)&base_address,
                    &alloc_length, (hipDeviceptr_t)ptr);
            if (hip_err != hipSuccess) {
                mc_error(&ucc_mc_rocm.super,
                         "hipMemGetAddressRange(%p) error: %d(%s)",
                          ptr, hip_err, hipGetErrorString(st));
                return UCC_ERR_NOT_SUPPORTED;
            }

            mem_attr->base_address = base_address;
            mem_attr->alloc_length = alloc_length;
        }
    }

    return UCC_OK;
}


ucc_status_t ucc_ee_rocm_task_post(void *ee_stream, void **ee_req)
{
    ucc_mc_rocm_stream_request_t *req;
    ucc_mc_rocm_event_t *rocm_event;
    ucc_status_t status;
    ucc_mc_rocm_config_t *cfg = MC_ROCM_CONFIG;

    req = ucc_mpool_get(&ucc_mc_rocm.strm_reqs);
    ucc_assert(req);
    req->status = UCC_MC_ROCM_TASK_POSTED;
    req->stream = (hipStream_t)ee_stream;

    if (ucc_mc_rocm.task_strm_type == UCC_MC_ROCM_USER_STREAM) {
        status = ucc_mc_rocm.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            req->stream);
        if (status != UCC_OK) {
            goto free_req;
        }
    } else {
        rocm_event = ucc_mpool_get(&ucc_mc_rocm.events);
        ucc_assert(rocm_event);
        ROCMCHECK(hipEventRecord(rocm_event->event, req->stream));
        ROCMCHECK(hipStreamWaitEvent(ucc_mc_rocm.stream, rocm_event->event, 0));
        status = ucc_mc_rocm.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            ucc_mc_rocm.stream);
        if (status != UCC_OK) {
            goto free_event;
        }
        ROCMCHECK(hipEventRecord(rocm_event->event, ucc_mc_rocm.stream));
        ROCMCHECK(hipStreamWaitEvent(req->stream, rocm_event->event, 0));
        ucc_mpool_put(rocm_event);
    }

    *ee_req = (void *) req;

    mc_info(&ucc_mc_rocm.super, "ROCM stream task posted on \"%s\" stream. req:%p",
            task_stream_types[ucc_mc_rocm.task_strm_type], req);

    return UCC_OK;

free_event:
    ucc_mpool_put(rocm_event);
free_req:
    ucc_mpool_put(req);
    return status;
}

ucc_status_t ucc_ee_rocm_task_query(void *ee_req)
{
    ucc_mc_rocm_stream_request_t *req = ee_req;

    /* ee task might be only in POSTED, STARTED or COMPLETED_ACK state
       COMPLETED state is used by ucc_ee_rocm_task_end function to request
       stream unblock*/
    ucc_assert(req->status != UCC_MC_ROCM_TASK_COMPLETED);
    if (req->status == UCC_MC_ROCM_TASK_POSTED) {
        return UCC_INPROGRESS;
    }
    mc_info(&ucc_mc_rocm.super, "ROCM stream task started. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_task_end(void *ee_req)
{
    ucc_mc_rocm_stream_request_t *req = ee_req;
    volatile ucc_mc_task_status_t *st = &req->status;

    /* can be safely ended only if it's in STARTED or COMPLETED_ACK state */
    ucc_assert((*st != UCC_MC_ROCM_TASK_POSTED) &&
               (*st != UCC_MC_ROCM_TASK_COMPLETED));
    if (*st == UCC_MC_ROCM_TASK_STARTED) {
        *st = UCC_MC_ROCM_TASK_COMPLETED;
        while(*st != UCC_MC_ROCM_TASK_COMPLETED_ACK) { }
    }
    ucc_mpool_put(req);
    mc_info(&ucc_mc_rocm.super, "ROCM stream task done. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_create_event(void **event)
{
    ucc_mc_rocm_event_t *rocm_event;

    rocm_event = ucc_mpool_get(&ucc_mc_rocm.events);
    ucc_assert(rocm_event);
    *event = rocm_event;
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_destroy_event(void *event)
{
    ucc_mc_rocm_event_t *rocm_event = event;

    ucc_mpool_put(rocm_event);
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_event_post(void *ee_context, void *event)
{
    hipStream_t stream = (hipStream_t )ee_context;
    ucc_mc_rocm_event_t *rocm_event = event;

    ROCMCHECK(hipEventRecord(rocm_event->event, stream));
    return UCC_OK;
}

ucc_status_t ucc_ee_rocm_event_test(void *event)
{
    hipError_t hip_err;
    ucc_mc_rocm_event_t *rocm_event = event;

    hip_err = hipEventQuery(rocm_event->event);
    return hip_error_to_ucc_status(hip_err);
}

ucc_mc_rocm_t ucc_mc_rocm = {
    .super.super.name       = "rocm mc",
    .super.ref_cnt          = 0,
    .super.type             = UCC_MEMORY_TYPE_ROCM,
    .super.init             = ucc_mc_rocm_init,
    .super.finalize         = ucc_mc_rocm_finalize,
    .super.ops.mem_query    = ucc_mc_rocm_mem_query,
    .super.ops.mem_alloc    = ucc_mc_rocm_mem_alloc,
    .super.ops.mem_free     = ucc_mc_rocm_mem_free,
    .super.ops.reduce       = NULL, //ucc_mc_rocm_reduce,
    .super.ops.reduce_multi = NULL, //ucc_mc_rocm_reduce_multi,
    .super.ops.memcpy       = ucc_mc_rocm_memcpy,
    .super.config_table =
        {
            .name   = "ROCM memory component",
            .prefix = "MC_ROCM_",
            .table  = ucc_mc_rocm_config_table,
            .size   = sizeof(ucc_mc_rocm_config_t),
        },
    .super.ee_ops.ee_task_post     = ucc_ee_rocm_task_post,
    .super.ee_ops.ee_task_query    = ucc_ee_rocm_task_query,
    .super.ee_ops.ee_task_end      = ucc_ee_rocm_task_end,
    .super.ee_ops.ee_create_event  = ucc_ee_rocm_create_event,
    .super.ee_ops.ee_destroy_event = ucc_ee_rocm_destroy_event,
    .super.ee_ops.ee_event_post    = ucc_ee_rocm_event_post,
    .super.ee_ops.ee_event_test    = ucc_ee_rocm_event_test,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_rocm.super.config_table,
                                &ucc_config_global_list);
