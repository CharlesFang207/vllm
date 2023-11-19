#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "dispatch_utils.h"

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace vllm {

template <typename scalar_t, typename int_t>
__device__ scalar_t bitwise_xor(scalar_t a, scalar_t b) {
    int_t a_int = *reinterpret_cast<int_t*>(&a);
    int_t b_int = *reinterpret_cast<int_t*>(&b);
    int_t result_int = a_int ^ b_int;
    return *reinterpret_cast<scalar_t*>(&result_int);
}

template <typename scalar_t>
__global__ void xor_kernel(
    scalar_t *dst,
    const scalar_t *src,
    const int64_t block_size_in_bytes) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < block_size_in_bytes / sizeof(scalar_t)) {
        dst[idx] = bitwise_xor<scalar_t, int>(dst[idx], src[idx]);
    }
}

} // namespace vllm

void xor_and_swap_out(
    torch::Tensor &src_1,
    torch::Tensor &dst_1,
    torch::Tensor &src_2,
    torch::Tensor &dst_2) {
    torch::Device src_1_device = src_1.device();
    torch::Device src_2_device = src_2.device();
    torch::Device dst_1_device = dst_1.device();
    torch::Device dst_2_device = dst_2.device();
    cudaMemcpyKind memcpy_type;

    // Copy KVCache from GPU 0 to GPU 1, do xor, then swap xor result to CPU
    if (src_1_device.is_cuda() && dst_1_device.is_cuda() && src_2_device.is_cuda() && dst_2_device.is_cpu()) {
        memcpy_type = cudaMemcpyDeviceToHost;
    } else {
        TORCH_CHECK(false, "swap_out_and_xor: unsupported device configuration");
    }

    void *src_1_ptr = src_1.data_ptr();
    void *src_2_ptr = src_2.data_ptr();
    void *dst_1_ptr = dst_1.data_ptr();
    void *dst_2_ptr = dst_2.data_ptr();

    const int64_t block_size_in_bytes = src_1.numel() * src_1.element_size();
    TORCH_CHECK(block_size_in_bytes == src_2.numel() * src_2.element_size());

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cudaMemcpyPeer(dst_1_ptr, dst_1_device.index(), src_1_ptr, src_1_device.index(), block_size_in_bytes);

    VLLM_DISPATCH_FLOATING_TYPES(
    src_1.scalar_type(), "xor_kernel", ([&] {
      vllm::xor_kernel<scalar_t><<<(block_size_in_bytes + 1024 - 1) / 1024, 1024, 0, stream>>>(
        reinterpret_cast<scalar_t *>(src_1_ptr), 
        reinterpret_cast<scalar_t *>(src_2_ptr), 
        block_size_in_bytes);
    }));

    cudaMemcpyAsync(dst_2_ptr, dst_1_ptr, block_size_in_bytes, memcpy_type, stream);

    cudaFree(dst_1_ptr);
}