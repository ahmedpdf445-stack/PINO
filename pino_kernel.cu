#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <stddef.h>

static inline int report_cuda(cudaError_t result) {
    return (int)result;
}

static inline int report_cufft(cufftResult result) {
    return 1000 + (int)result;
}

__global__ void navier_stokes_step_kernel(const float* state, float* next_state, int width, int height, float dt, float viscosity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    int xm = x == 0 ? width - 1 : x - 1;
    int xp = x + 1 == width ? 0 : x + 1;
    int ym = y == 0 ? height - 1 : y - 1;
    int yp = y + 1 == height ? 0 : y + 1;

    float center = state[idx];
    float laplacian = state[y * width + xp] + state[y * width + xm] + state[yp * width + x] + state[ym * width + x] - 4.0f * center;
    float diffusion = viscosity * laplacian;
    float gradient_x = 0.5f * (state[y * width + xp] - state[y * width + xm]);
    float gradient_y = 0.5f * (state[yp * width + x] - state[ym * width + x]);
    float advect = center * (gradient_x + gradient_y) * 0.1f;
    next_state[idx] = center + dt * (diffusion - advect);
}

__global__ void spectral_damp_kernel(cufftComplex* freq, int width, int height, float viscosity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = width / 2 + 1;
    if (x >= half || y >= height) {
        return;
    }

    int idx = y * half + x;
    float kx = (float)x;
    if (x > width / 2) {
        kx = (float)(x - width);
    }
    float ky = (float)y;
    if (y > height / 2) {
        ky = (float)(y - height);
    }
    float k2 = kx * kx + ky * ky;
    float damping = expf(-0.03f * k2) * (1.0f - 0.05f * viscosity);
    freq[idx].x *= damping;
    freq[idx].y *= damping;
}

__global__ void normalize_kernel(float* data, int width, int height, float scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int idx = y * width + x;
    data[idx] *= scale;
}

extern "C" {

int preflight_validate(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 1;
    }
    if (width % 2 != 0 || height % 2 != 0) {
        return 2;
    }
    if (width < 8 || height < 8) {
        return 3;
    }
    if (width > 2048 || height > 2048) {
        return 4;
    }
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        return report_cuda(err);
    }
    return 0;
}

void* allocate_zero_copy_buffer(size_t count) {
    void* ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, count * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

int free_zero_copy_buffer(void* ptr) {
    if (!ptr) {
        return 1;
    }
    cudaError_t err = cudaFreeHost(ptr);
    return (err == cudaSuccess) ? 0 : report_cuda(err);
}

int physics_loss_kernel(void* state_buffer, void* next_buffer, void* scratch_buffer, int width, int height, float dt, float viscosity) {
    if (!state_buffer || !next_buffer || !scratch_buffer) {
        return 1;
    }
    if (width <= 0 || height <= 0) {
        return 2;
    }
    if (dt <= 0.0f || viscosity < 0.0f) {
        return 3;
    }
    if (width % 2 != 0 || height % 2 != 0) {
        return 4;
    }

    float* device_state = nullptr;
    float* device_next = nullptr;
    cufftComplex* device_freq = nullptr;
    cudaError_t cuda_err = cudaHostGetDevicePointer((void**)&device_state, state_buffer, 0);
    if (cuda_err != cudaSuccess) {
        return report_cuda(cuda_err);
    }
    cuda_err = cudaHostGetDevicePointer((void**)&device_next, next_buffer, 0);
    if (cuda_err != cudaSuccess) {
        return report_cuda(cuda_err);
    }
    cuda_err = cudaHostGetDevicePointer((void**)&device_freq, scratch_buffer, 0);
    if (cuda_err != cudaSuccess) {
        return report_cuda(cuda_err);
    }

    size_t size = static_cast<size_t>(width) * static_cast<size_t>(height);
    int half = width / 2 + 1;
    cufftHandle plan = 0;
    cufftResult cufft_err = cufftPlan2d(&plan, height, width, CUFFT_R2C);
    if (cufft_err != CUFFT_SUCCESS) {
        return report_cufft(cufft_err);
    }

    cufft_err = cufftExecR2C(plan, device_state, device_freq);
    if (cufft_err != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        return report_cufft(cufft_err);
    }

    dim3 block(16, 16);
    dim3 grid((half + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    spectral_damp_kernel<<<grid, block>>>(device_freq, width, height, viscosity);
    cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        cufftDestroy(plan);
        return report_cuda(cuda_err);
    }

    cufft_err = cufftExecC2R(plan, device_freq, device_next);
    if (cufft_err != CUFFT_SUCCESS) {
        cufftDestroy(plan);
        return report_cufft(cufft_err);
    }

    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        cufftDestroy(plan);
        return report_cuda(cuda_err);
    }

    grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    normalize_kernel<<<grid, block>>>(device_next, width, height, 1.0f / static_cast<float>(size));
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        cufftDestroy(plan);
        return report_cuda(cuda_err);
    }

    navier_stokes_step_kernel<<<grid, block>>>(device_state, device_next, width, height, dt, viscosity);
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        cufftDestroy(plan);
        return report_cuda(cuda_err);
    }

    cufftDestroy(plan);
    return 0;
}

}
