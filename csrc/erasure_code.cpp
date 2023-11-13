#include <torch/extension.h>

#include <map>
#include <vector>

void xor_and_swap_out(
    torch::Tensor &src_1,
    torch::Tensor &dst_1,
    torch::Tensor &src_2,
    torch::Tensor &dst_2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "xor_and_swap_out",
      &xor_and_swap_out,
      "do xor for KV cache from two devices and swap out");
}
