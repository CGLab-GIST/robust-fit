//  Copyright(c) 2019 CGLab, GIST. All rights reserved.
//  
//  Redistribution and use in source and binary forms, with or without modification, 
//  are permitted provided that the following conditions are met :
// 
//  - Redistributions of source code must retain the above copyright notice, 
//    this list of conditions and the following disclaimer.
//  - Redistributions in binary form must reproduce the above copyright notice, 
//    this list of conditions and the following disclaimer in the documentation
//    and / or other materials provided with the distribution.
//  - Neither the name of the copyright holder nor the names of its contributors 
//    may be used to endorse or promote products derived from this software 
//    without specific prior written permission.
//  
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
//  ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
//  DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
//  OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"
#include "helper_math.h"

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h> 
#include <thrust/functional.h> 
#include <thrust/device_vector.h> 
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "RobustFitOptions.h"
#include "RobustFitCUDA.h"
#include "RobustFitMem.h"
#include "RobustFitUtilCUDA.h"

#include <time.h>

#define globalThreadIdx (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y)))

// For buffers
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_tp;
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_dx;
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_dy;
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_varDx;
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_varDy;

cudaArray *g_src_tp;
cudaArray *g_src_dx;
cudaArray *g_src_dy;
cudaArray *g_src_varDx;
cudaArray *g_src_varDy;

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

inline int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

template <typename T>
struct square {
	__host__ __device__
		T operator()(const T& x) const {
		return x * x;
	}
};

__global__ void kernel_adjust_weight(float* _w2, const float nEle, const float* sum_w2, const int xSize, const int ySize, float alpha) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int nPix = xSize * ySize;
	const int i = cy * xSize + cx;

	float factor = nEle / sum_w2[0];
	_w2[nPix * 0 + i] *= factor;
	_w2[nPix * 1 + i] *= factor;
	_w2[nPix * 2 + i] *= factor;
}

__global__ void kernel_update_edge(char2* _edges, const PixError* _pixErr, const int nEdges, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int n = xSize * ySize;
	const int i = 2 * (cy * xSize + cx);

	if (i < nEdges) {
		int idx = _pixErr[i].idx_grad;
		if (idx < n)
			_edges[idx].x = 1;
		else
			_edges[idx % n].y = 1;
	}

	if (i + 1 < nEdges) {
		int idx = _pixErr[i + 1].idx_grad;
		if (idx < n)
			_edges[idx].x = 1;
		else
			_edges[idx % n].y = 1;
	}
}

__global__ void kernel_calc_PTW2x(float4* _rr_old, float4* _r, const float* _w2, const float4* _e, const int xSize, const int ySize, float alpha) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int n = xSize * ySize;
	const int i = cy * xSize + cx;

	float4 PTW2xi = _w2[i] * _e[i] * alpha;
	if (cx != 0)
		PTW2xi += _w2[n + i - 1] * _e[n + i - 1];
	if (cx != xSize - 1)
		PTW2xi -= _w2[n + i] * _e[n + i];
	if (cy != 0)
		PTW2xi += _w2[2 * n + i - xSize] * _e[2 * n + i - xSize];
	if (cy != ySize - 1)
		PTW2xi -= _w2[2 * n + i] * _e[2 * n + i];
	_r[i] = PTW2xi;

	atomicAdd(&_rr_old[0].x, _r[i].x * _r[i].x);
	atomicAdd(&_rr_old[0].y, _r[i].y * _r[i].y);
	atomicAdd(&_rr_old[0].z, _r[i].z * _r[i].z);
}

__global__ void kernel_update_x(float4* _x, const float4* _p, const float4 alpha, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	_x[i] += alpha * _p[i];
}

__global__ void kernel_update_x_p(float4* _x, float4* _p, const float4* _r, const float4* alpha, const float4 *beta, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;

	float4 _pi = _p[i];
	_x[i] += alpha[0] * _pi;
	_p[i] = _r[i] + beta[0] * _p[i];
}

__global__ void kernel_calc_Ap_pAp(float4* pAp_sum, float4* pAp, float4* Ap, const float* _w2, const float4* _x, const int xSize, const int ySize, const float alpha) {

	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;

	const int n = xSize * ySize;
	const int idx = cy * xSize + cx;

	const float4& xi = _x[idx];
	float4 PTW2xi = _w2[idx] * xi * alpha * alpha;
	if (cx != 0)         PTW2xi += _w2[n + idx - 1] * (xi - _x[idx - 1]);
	if (cx != xSize - 1) PTW2xi += _w2[n + idx] * (xi - _x[idx + 1]);
	if (cy != 0)         PTW2xi += _w2[2 * n + idx - xSize] * (xi - _x[idx - xSize]);
	if (cy != ySize - 1) PTW2xi += _w2[2 * n + idx] * (xi - _x[idx + xSize]);

	Ap[idx].x = PTW2xi.x;
	Ap[idx].y = PTW2xi.y;
	Ap[idx].z = PTW2xi.z;

	pAp[idx].x = _x[idx].x * Ap[idx].x;
	pAp[idx].y = _x[idx].y * Ap[idx].y;
	pAp[idx].z = _x[idx].z * Ap[idx].z;

	atomicAdd(&pAp_sum[0].x, pAp[idx].x);
	atomicAdd(&pAp_sum[0].y, pAp[idx].y);
	atomicAdd(&pAp_sum[0].z, pAp[idx].z);
}

__global__ void kernel_update_sum_rr(float4* _sum_rr_old, float4* _sum_rr_new) {
	_sum_rr_old[0] = _sum_rr_new[0];
	_sum_rr_new[0].x = FLT_MIN;
	_sum_rr_new[0].y = FLT_MIN;
	_sum_rr_new[0].z = FLT_MIN;
	_sum_rr_new[0].w = FLT_MIN;
}

__global__ void kernel_divide(float4* factor, float4* num, float4* denom) {
	factor[0] = num[0] / max(denom[0], FLT_MIN);
}

__global__ void kernel_calc_r(float4* _r_sum, float4* _r, const float4* _Ap, const float4* alpha, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	int i = cy * xSize + cx;
	_r[i] = _r[i] - alpha[0] * _Ap[i];

	atomicAdd(&_r_sum[0].x, _r[i].x * _r[i].x);
	atomicAdd(&_r_sum[0].y, _r[i].y * _r[i].y);
	atomicAdd(&_r_sum[0].z, _r[i].z * _r[i].z);
}

__global__ void kernel_calc_Ax(float4* _out, const float4* _x, const int xSize, const int ySize, const float alpha) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int n = xSize * ySize;
	const int idx = cy * xSize + cx;

	_out[idx] = alpha * _x[idx];
	_out[n + idx].x = (cx != xSize - 1) ? (_x[idx + 1].x - _x[idx].x) : 0.f;
	_out[n + idx].y = (cx != xSize - 1) ? (_x[idx + 1].y - _x[idx].y) : 0.f;
	_out[n + idx].z = (cx != xSize - 1) ? (_x[idx + 1].z - _x[idx].z) : 0.f;

	_out[n * 2 + idx].x = (cy != ySize - 1) ? (_x[idx + xSize].x - _x[idx].x) : 0.f;
	_out[n * 2 + idx].y = (cy != ySize - 1) ? (_x[idx + xSize].y - _x[idx].y) : 0.f;
	_out[n * 2 + idx].z = (cy != ySize - 1) ? (_x[idx + xSize].z - _x[idx].z) : 0.f;
}

__global__ void kernel_calc_residual(float4* _e, const float4* _b, const float4* _Ax, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int n = xSize * ySize;
	const int i = cy * xSize + cx;

	_e[i] = _b[i] - _Ax[i];
	_e[n + i] = _b[n + i] - _Ax[n + i];
	_e[n * 2 + i] = _b[2 * n + i] - _Ax[2 * n + i];
}

__global__ void kernel_set_weights(float* _w2, const char2* _edges, const int xSize, const int ySize, const float alpha) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int nPix = xSize * ySize;
	const int i = cy * xSize + cx;

	_w2[nPix * 0 + i] = 1.f;
	_w2[nPix * 1 + i] = (_edges[i].x == 1) ? 1.f : 0.f;
	_w2[nPix * 2 + i] = (_edges[i].y == 1) ? 1.f : 0.f;
}

__global__ void kernel_calc_wgt2(float* _sum_w2, float* _w2, float4* _e, const char2* _edges, float reg, const int xSize, const int ySize, const float alpha, bool isLast) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int nPix = xSize * ySize;
	const int i = cy * xSize + cx;

	int idxEle;
	float length_e;
	float w2i;

	idxEle = nPix * 0 + i;
	length_e = sqrtf(_e[idxEle].x * _e[idxEle].x + _e[idxEle].y * _e[idxEle].y + _e[idxEle].z * _e[idxEle].z);
	w2i = 1.0f / (length_e + reg);
	_w2[idxEle] = w2i;
	atomicAdd(&_sum_w2[0], w2i);

	idxEle = nPix * 1 + i;
	length_e = sqrtf(_e[idxEle].x * _e[idxEle].x + _e[idxEle].y * _e[idxEle].y + _e[idxEle].z * _e[idxEle].z);
	w2i = (_edges[i].x == 1) ? 1.0f / (length_e + reg) : 0.f;
	_w2[idxEle] = w2i;
	atomicAdd(&_sum_w2[0], w2i);

	idxEle = nPix * 2 + i;
	length_e = sqrtf(_e[idxEle].x * _e[idxEle].x + _e[idxEle].y * _e[idxEle].y + _e[idxEle].z * _e[idxEle].z);
	w2i = (_edges[i].y == 1) ? 1.0f / (length_e + reg) : 0.f;
	_w2[idxEle] = w2i;
	atomicAdd(&_sum_w2[0], w2i);
}

__global__ void kernel_setup_b(float4* _y, const int xSize, const int ySize, const float alpha) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int nPix = xSize * ySize;
	const int i = cy * xSize + cx;
	_y[nPix * 0 + i] = alpha * tex2D(g_tp, cx, cy);
	_y[nPix * 1 + i] = tex2D(g_dx, cx, cy);
	_y[nPix * 2 + i] = tex2D(g_dy, cx, cy);
}

__global__ void kernel_add_edges(float* _key_sort, PixError* _pixErr, const char2* _edgesMST, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int n = xSize * ySize;
	const int i = cy * xSize + cx;

	const float4& varDx = tex2D(g_varDx, cx, cy);
	float key_sort = (_edgesMST[i].x == 1) ? -1e10f : (fabs(varDx.x) + fabs(varDx.y) + fabs(varDx.z)) / 3.f;
	_key_sort[i] = (cx != xSize - 1) ? key_sort : 1e10f;
	_pixErr[i].error = 0.f;
	_pixErr[i].idx_grad = i;

	const float4& varDy = tex2D(g_varDy, cx, cy);
	key_sort = (_edgesMST[i].y == 1) ? -1e10f : (fabs(varDy.x) + fabs(varDy.y) + fabs(varDy.z)) / 3.f;
	_key_sort[n + i] = (cy != ySize - 1) ? key_sort : 1e10f;
	_pixErr[n + i].error = 0.f;
	_pixErr[n + i].idx_grad = n + i;
}

__global__ void kernel_estimate_err(double *err_sum, const float4* _img, const float4* _nlm, const int xSize, const int ySize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;

	const int i = cy * xSize + cx;
	const int nPix = xSize * ySize;
	const double epsilon = 0.01;

	double l2err = (_img[i].x - _nlm[i].x) * (_img[i].x - _nlm[i].x) +
		(_img[i].y - _nlm[i].y) * (_img[i].y - _nlm[i].y) +
		(_img[i].z - _nlm[i].z) * (_img[i].z - _nlm[i].z);
	double lum = (_nlm[i].x + _nlm[i].y + _nlm[i].z) / 3.0;
	double relErr = l2err / (lum * lum + epsilon);

	atomicAdd(&err_sum[0], relErr);
}

void RobustFitMem::allocMemory(int width, int height) {
	int nPix = width * height;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	checkCudaErrors(cudaMalloc((void **)&_d_key_sort, 2 * nPix * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&_d_pixErr, 2 * nPix * sizeof(PixError)));
	checkCudaErrors(cudaMalloc((void **)&_d_edges_MST, nPix * sizeof(char2)));
	checkCudaErrors(cudaMalloc((void **)&_d_edges, nPix * sizeof(char2)));
	checkCudaErrors(cudaMalloc((void **)&_d_optEdges, nPix * sizeof(char2)));

	checkCudaErrors(cudaMallocArray(&g_src_tp, &channelDesc, width, height));
	checkCudaErrors(cudaMallocArray(&g_src_dx, &channelDesc, width, height));
	checkCudaErrors(cudaMallocArray(&g_src_dy, &channelDesc, width, height));
	checkCudaErrors(cudaMallocArray(&g_src_varDx, &channelDesc, width, height));
	checkCudaErrors(cudaMallocArray(&g_src_varDy, &channelDesc, width, height));

	checkCudaErrors(cudaGetLastError());
}

void RobustFitMem::deleteMemory() {
	checkCudaErrors(cudaFree(_d_key_sort));
	checkCudaErrors(cudaFree(_d_pixErr));
	checkCudaErrors(cudaFree(_d_edges_MST));
	checkCudaErrors(cudaFree(_d_edges));
	checkCudaErrors(cudaFree(_d_optEdges));

	checkCudaErrors(cudaFreeArray(g_src_tp));
	checkCudaErrors(cudaFreeArray(g_src_dx));
	checkCudaErrors(cudaFreeArray(g_src_dy));
	checkCudaErrors(cudaFreeArray(g_src_varDx));
	checkCudaErrors(cudaFreeArray(g_src_varDy));

	checkCudaErrors(cudaGetLastError());
}

float adjustSolverAlpha(float initAlpha, int numEdges, int maxEdges) {
	return (initAlpha);
}

// (W*P)'*W*P*x = (W*P)'*W*b
// (P'*W^2*P)*x = (P'*W^2*b)
// A*x = bb,
// where
//      A  = P'*W^2*P = P'*diag(w2)*P = NxN matrix
//      bb = P'*W^2*b = P'*diag(w2)*b = Nx1 vector
//      w2 = (N*3)x1 vector of squared weights

void run_conjugate_gradient_gpu(SolverConfig solverConfig, float alpha, SolverGPUMem& gpuMem, const char2* _edges, int nEdge, int xSize, int ySize) {
	// Description  : changed CG & IRLS and run completely seperately

	int nPix = xSize * ySize;
	int nEle = 3 * nPix;
	const int blockDim = 32;

	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));

	float *sum_W2;

	checkCudaErrors(cudaMalloc((void **)&sum_W2, sizeof(float)));

	int IRLS_MAX_ITER = solverConfig.irls_max_iter;
	float IRLS_REG_INIT = solverConfig.irls_reg_init;
	float IRLS_REG_ITER = solverConfig.irls_reg_iter;
	int CG_MAX_ITER = solverConfig.cg_max_iter;

	kernel_setup_b<< <grid, threads >> > (gpuMem._b, xSize, ySize, alpha);

	for (int irlsIter = 0; irlsIter < IRLS_MAX_ITER; ++irlsIter) {

		checkCudaErrors(cudaMemset(sum_W2, 0, sizeof(float)));
		checkCudaErrors(cudaMemset(gpuMem._sum_rr_old, 0, sizeof(float4)));

		// Ax
		kernel_calc_Ax << <grid, threads >> > (gpuMem._Ax, gpuMem._x, xSize, ySize, alpha);

		// b-Ax
		kernel_calc_residual << <grid, threads >> > (gpuMem._e, gpuMem._b, gpuMem._Ax, xSize, ySize);
	
		if (irlsIter == 0) {
			kernel_set_weights << <grid, threads >> >(gpuMem._w2, _edges, xSize, ySize, alpha);
		}
		else {
			// w2 = coef / (length(e) + reg)
			float reg = IRLS_REG_INIT * powf(IRLS_REG_ITER, (float)(irlsIter - 1));
			bool isLast = (irlsIter == IRLS_MAX_ITER - 1) ? true : false;

			kernel_calc_wgt2 << <grid, threads >> > (sum_W2, gpuMem._w2, gpuMem._e, _edges, reg, xSize, ySize, alpha, isLast);
			checkCudaErrors(cudaDeviceSynchronize());

			// Weight Normalization 
			kernel_adjust_weight << <grid, threads >> > (gpuMem._w2, (float)(nPix + nEdge), sum_W2, xSize, ySize, alpha);
		}

		// r = PtW(b - Px)
		kernel_calc_PTW2x << <grid, threads >> > (gpuMem._sum_rr_old, gpuMem._r, gpuMem._w2, gpuMem._e, xSize, ySize, alpha);
		// p = r
		checkCudaErrors(cudaMemcpy(gpuMem._p, gpuMem._r, nPix * sizeof(float4), cudaMemcpyDeviceToDevice));
		
		const float tol = 1e-10f;

		for (int cgIter = 0; cgIter < CG_MAX_ITER; ++cgIter) {

			checkCudaErrors(cudaMemset(gpuMem._sum_pq, 0, sizeof(float4)));
			
			// Ap = A*p = AtW2Ap
			// pAp = p'*A*p
			kernel_calc_Ap_pAp << <grid, threads >> > (gpuMem._sum_pq, gpuMem._pq, gpuMem._AtW2Ap, gpuMem._w2, gpuMem._p, xSize, ySize, alpha);
			kernel_divide << <1, 1 >> > (gpuMem._alpha, gpuMem._sum_rr_old, gpuMem._sum_pq);
			
			// r -= Ap*(rz2/pAp)
			kernel_calc_r << <grid, threads >> > (gpuMem._sum_rr_new, gpuMem._r, gpuMem._AtW2Ap, gpuMem._alpha, xSize, ySize);

			kernel_divide << <1, 1 >> > (gpuMem._beta, gpuMem._sum_rr_new, gpuMem._sum_rr_old);
			
			// x += p*(rz2/pAp)
			// p = r + p*(rz/rz2)
			kernel_update_x_p << <grid, threads >> > (gpuMem._x, gpuMem._p, gpuMem._r, gpuMem._alpha, gpuMem._beta, xSize, ySize);
			kernel_update_sum_rr << <1, 1 >> > (gpuMem._sum_rr_old, gpuMem._sum_rr_new);
		}
	}

}

void add_edges(RobustFitMem& cudaMem, int xSize, int ySize)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	const int nPix = xSize * ySize;
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));

	const int MAX_NUM_EDGES = 2 * nPix - xSize - ySize;

	kernel_add_edges << <grid, threads >> > (cudaMem._d_key_sort, cudaMem._d_pixErr, cudaMem._d_edges_MST, xSize, ySize);
	checkCudaErrors(cudaDeviceSynchronize());

	thrust::sort_by_key(thrust::device, cudaMem._d_key_sort, cudaMem._d_key_sort + nPix * 2, cudaMem._d_pixErr);
}

extern "C" void robust_fit_cuda(SolverConfig solver_config, float initAlpha, char2* _edges,
	const float4* _tp, const float4* _dx, const float4* _dy, const float4* _varDx, const float4* _varDy,
	const float4* _nlmImg, int xSize, int ySize, const int nInitEdges,
	float4* _optImg)
{


	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	SolverConfig L2SolverConfig;
	L2SolverConfig.irls_max_iter = 1;
	L2SolverConfig.irls_reg_init = 0.f;
	L2SolverConfig.irls_reg_iter = 0.f;
	L2SolverConfig.cg_max_iter = 50;

	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	const int nPix = xSize * ySize;
	const int NUM_MST_EDGES = nInitEdges;
	const int MAX_NUM_EDGES = 2 * nPix - xSize - ySize;
	// 1. Memory Setup - this should be done one time if this function is performed multiple times (e.g., animation)
	float4 *_output = (float4*)malloc(sizeof(float4) * nPix);

	float4 *_d_nlm;
	checkCudaErrors(cudaMalloc((void **)&_d_nlm, nPix * sizeof(float4)));
	checkCudaErrors(cudaMemcpy(_d_nlm, _nlmImg, nPix * sizeof(float4), cudaMemcpyHostToDevice));

	RobustFitMem cudaMem;
	cudaMem.allocMemory(xSize, ySize);

	checkCudaErrors(cudaMemcpyToArray(g_src_tp, 0, 0, _tp, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_dx, 0, 0, _dx, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_dy, 0, 0, _dy, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_varDx, 0, 0, _varDx, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_varDy, 0, 0, _varDy, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTextureToArray(&g_tp, g_src_tp, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_dx, g_src_dx, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_dy, g_src_dy, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_varDx, g_src_varDx, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_varDy, g_src_varDy, &channelDesc));

	checkCudaErrors(cudaMemcpy(cudaMem._d_edges_MST, _edges, nPix * sizeof(char2), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(cudaMem._d_edges, cudaMem._d_edges_MST, nPix * sizeof(char2), cudaMemcpyDeviceToDevice));

	SolverGPUMem solverMem = SolverGPUMem(nPix);
	checkCudaErrors(cudaMemcpy(solverMem._x, _tp, nPix * sizeof(float4), cudaMemcpyHostToDevice));

	float alpha = adjustSolverAlpha(initAlpha, NUM_MST_EDGES, MAX_NUM_EDGES);
	int optEdges = 0;

	double *_d_err_sum;
	checkCudaErrors(cudaMalloc((void **)&_d_err_sum, sizeof(double)));
	double *err_sum = (double*)malloc(sizeof(double));
	double optErr = 10.0;

	const int nEdgesStart = (int)ceilf(MAX_NUM_EDGES * 0.01f * (float)ITER_START_PERCENT);
	const int nEdgesPerStep = (int)ceilf((MAX_NUM_EDGES - nEdgesStart) / (float)(ITER_AUGMENTATION - 1));

#ifdef DEBUG_SOLVER  
	clock_t sTime, eTime;
	double elapsed_secs_total = 0.0;
	sTime = clock();
#endif

	add_edges(cudaMem, xSize, ySize);

	// 2. Edge Augmentation (gradient subset updates)
	for (int iter = 0; iter < ITER_AUGMENTATION; ++iter) {
		int nEdges = min(MAX_NUM_EDGES, nEdgesStart + nEdgesPerStep * (iter));
		checkCudaErrors(cudaMemset(_d_err_sum, 0.0, sizeof(double)));

		checkCudaErrors(cudaMemset(cudaMem._d_edges, 0, nPix * sizeof(char2)));
		checkCudaErrors(cudaMemcpy(solverMem._x, _tp, nPix * sizeof(float4), cudaMemcpyHostToDevice));

		// Augment edges and update edge map (augment the number of gradients in a subset)
		kernel_update_edge << <grid, threads >> > (cudaMem._d_edges, cudaMem._d_pixErr, nEdges, xSize, ySize);
		checkCudaErrors(cudaDeviceSynchronize());

		// Run CG and estimate reconstruction error
		float alpha = adjustSolverAlpha(initAlpha, nEdges, MAX_NUM_EDGES);
		run_conjugate_gradient_gpu(L2SolverConfig, alpha, solverMem, cudaMem._d_edges, nEdges, xSize, ySize);
		kernel_estimate_err << <grid, threads >> > (_d_err_sum, solverMem._x, _d_nlm, xSize, ySize); // reconstruction error estimation
		checkCudaErrors(cudaMemcpy(err_sum, _d_err_sum, sizeof(double), cudaMemcpyDeviceToHost));
		double err = *err_sum / ((double)nPix * 3.0);

		// If the rconstruction error is nimimum, the current edge map 
		// which indicates each gradient should be included or not 
		// is updated as the optimal one.
		if (optErr > err) {
			optErr = err;
			optEdges = nEdges;
			checkCudaErrors(cudaMemcpy(cudaMem._d_optEdges, cudaMem._d_edges, nPix * sizeof(char2), cudaMemcpyDeviceToDevice));
		}
	}
#ifdef DEBUG_SOLVER  
	eTime = clock();
	elapsed_secs_total = (double)(eTime - sTime);
	printf("Augmentation Time (Total) = %.4f secs\n", elapsed_secs_total / CLOCKS_PER_SEC);
#endif
	checkCudaErrors(cudaMemset(cudaMem._d_edges, 0, nPix * sizeof(char2)));
	checkCudaErrors(cudaMemcpy(solverMem._x, _tp, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	// final CG using the optimal edge map 
	run_conjugate_gradient_gpu(solver_config, alpha, solverMem, cudaMem._d_optEdges, optEdges, xSize, ySize);

	checkCudaErrors(cudaMemcpy(_output, solverMem._x, nPix * sizeof(float4), cudaMemcpyDeviceToHost));
	memcpy(_optImg, _output, sizeof(float4) * nPix);

	checkCudaErrors(cudaUnbindTexture(&g_tp));
	checkCudaErrors(cudaUnbindTexture(&g_dx));
	checkCudaErrors(cudaUnbindTexture(&g_dy));
	checkCudaErrors(cudaUnbindTexture(&g_varDx));
	checkCudaErrors(cudaUnbindTexture(&g_varDy));
	checkCudaErrors(cudaGetLastError());

	delete[] _output;
}