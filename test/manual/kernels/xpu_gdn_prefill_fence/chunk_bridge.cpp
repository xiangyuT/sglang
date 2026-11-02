// Diagnostic replacement of the native BF16 / FP32-state chunk dispatcher.
// Preserve the installed convolution and preprocessing implementation.
#include <atomic>
#include <torch/all.h>
#define SGL_GDN_JIT_ENTRY
#define gdn gguf_gdn_fence_probe
#include "sycl/kernels/gdn_attn/chunk_gated_delta_rule_kernels.hpp"
#include "sycl/kernels/gdn_attn/chunk_compute_A_inverse_fused.hpp"
#undef gdn
static std::atomic<uint64_t> calls{0};
extern "C" uint64_t gguf_gdn_fence_calls(){return calls.load();}

__attribute__((visibility("default"))) void chunk_gated_delta_rule(
    sycl::queue& queue, torch::Tensor& core,
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& b, const torch::Tensor& a,
    const torch::Tensor& alog, const torch::Tensor& dt,
    torch::Tensor& ssm, const torch::Tensor& positions,
    const torch::Tensor& indices, const std::optional<torch::Tensor>& initial,
    int num_prefills, int num_decodes, const int* token_indices) {
  TORCH_CHECK(core.scalar_type()==at::kBFloat16 && ssm.scalar_type()==at::kFloat,
              "Diagnostic dispatcher supports BF16 activation and FP32 state only");
  ++calls;
  int batch=positions.size(0)-1;
  int n=core.size(0)+batch*63;
  auto options=core.options();
  auto A=torch::zeros({v.size(1),n,64},options);
  auto w=torch::zeros({v.size(1),n,q.size(2)},options);
  auto u=torch::zeros({v.size(1),n,v.size(2)},options);
  auto chunks=torch::zeros({1},options.dtype(torch::kInt32));
  gguf_gdn_fence_probe::kernel_launcher<cute::bfloat16_t,float>(
    queue,reinterpret_cast<cute::bfloat16_t*>(core.data_ptr()),
    reinterpret_cast<const cute::bfloat16_t*>(q.data_ptr()),
    reinterpret_cast<const cute::bfloat16_t*>(k.data_ptr()),
    reinterpret_cast<const cute::bfloat16_t*>(v.data_ptr()),
    reinterpret_cast<cute::bfloat16_t*>(A.data_ptr()),
    reinterpret_cast<cute::bfloat16_t*>(w.data_ptr()),
    reinterpret_cast<cute::bfloat16_t*>(u.data_ptr()),chunks.data_ptr<int>(),
    b.data_ptr<float>(),a.data_ptr<float>(),alog.data_ptr<float>(),
    reinterpret_cast<const cute::bfloat16_t*>(dt.data_ptr()),ssm.data_ptr<float>(),
    ssm.stride(0),positions.data_ptr<int>(),indices.data_ptr<int>(),
    initial.has_value()?initial->data_ptr<bool>():nullptr,token_indices,
    batch,q.size(0),q.size(1),q.size(2),v.size(1),v.size(2));
}
