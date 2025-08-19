#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"
#include <cmath>
#include <cuda_runtime.h>

namespace infini_train::kernels::cuda {

__global__ void AccumulateGradKernel(const float *grad_ptr, float rate, float *tensor_ptr, size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        tensor_ptr[idx] += rate * grad_ptr[idx];
    }
}

void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    size_t num_elements = gradient->NumElements();

    const float *grad_ptr = static_cast<const float *>(gradient->DataPtr());
    float *tensor_ptr = static_cast<float *>(tensor->DataPtr());

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, rate, tensor_ptr, num_elements);
}
__global__ void AdamAccumulateKernel(const float *grad_ptr, float *param_ptr,
                                     float *m_ptr, float *v_ptr, float learning_rate,
                                     float beta1, float beta2, float eps,
                                     float inv_one_minus_beta1,
                                     float inv_one_minus_beta2,
                                     size_t num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {

        float g = grad_ptr[idx];
        float m = m_ptr[idx];
        float v = v_ptr[idx];
        float param = param_ptr[idx];


        m = beta1 * m + (1 - beta1) * g;
        v = beta2 * v + (1 - beta2) * g * g;

        m_ptr[idx] = m;
        v_ptr[idx] = v;

        float m_hat = m * inv_one_minus_beta1;
        float v_hat = v * inv_one_minus_beta2;

        param_ptr[idx] = param - learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // FINISHED：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    size_t num_elements = grad->NumElements();

    const float *grad_ptr = static_cast<const float *>(grad->DataPtr());
    float *param_ptr = static_cast<float *>(param->DataPtr());
    float *m_ptr = static_cast<float *>(m->DataPtr());
    float *v_ptr = static_cast<float *>(v->DataPtr());

    float beta1_power = powf(beta1, static_cast<float>(t));
    float inv_one_minus_beta1 = 1.f / (1.f - beta1_power);
    float beta2_power = powf(beta2, static_cast<float>(t));
    float inv_one_minus_beta2 = 1.f / (1.f - beta2_power);

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    AdamAccumulateKernel<<<num_blocks, threads_per_block>>>(
        grad_ptr, param_ptr, m_ptr, v_ptr, learning_rate,
        beta1, beta2, eps, inv_one_minus_beta1, inv_one_minus_beta2, num_elements);
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                              \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CUDA_ACCUMULATE_GRAD_KERNEL
