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

#include "FilterCUDA.h"
#include "RobustFitOptions.h"
#include "RobustFitUtilCUDA.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"
#include "helper_math.h"

// From Rousselle's NLM
#include <algorithm>
using std::fill;
#include <numeric>
using std::accumulate;
#include <cmath>
using std::exp;
#include <iostream>
using std::cout;
using std::endl;
#include <omp.h>

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

inline int iDivUp(int a, int b) { return ((a % b) != 0) ? (a / b + 1) : (a / b); }

texture<float4, cudaTextureType2D, cudaReadModeElementType> g_img;
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_imgVar;
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_dx;
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_dy;

__device__ float4 calc_color_distance(const float4& a, const float4& b, const float4& varA, const float4& varB, float ep, float k) {
	float4 ret = (a - b) * (a - b) - (varA + varB);
	float4 denominator = (make_float4(ep) + k * k * (varA + varB));
	ret /= denominator;
	return ret;
}

__device__ float calc_patch_weight(float4& dist2, int patchRadius) {
	int nEle = 3 * (2 * patchRadius + 1) * (2 * patchRadius + 1);
	float avgDist2 = fmaxf(0.f, (dist2.x + dist2.y + dist2.z) / (float)nEle);
	return (expf(-avgDist2));
}

__global__ void kernel_filter_nlm(float4* _outImg, float4* _outDx, float4* _outDy, int xSize, int ySize, const int HALF_WINDOW, const float kc, const int tc) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= xSize || cy >= ySize)
		return;
	const int cIdx = cy * xSize + cx;
	const float epsilon = 1e-10f;

	int2 sWindow = make_int2(cx - HALF_WINDOW, cy - HALF_WINDOW);
	int2 eWindow = make_int2(cx + HALF_WINDOW, cy + HALF_WINDOW);

	float sumW = 0.f;
	float4 outImg = make_float4(0.f);
	float4 outDx = make_float4(0.f);
	float4 outDy = make_float4(0.f);

	for (int y = sWindow.y; y <= eWindow.y; ++y) {
		for (int x = sWindow.x; x <= eWindow.x; ++x) {
			// patchwise distance for the color
			float4 dist_col = make_float4(0.f);
			for (int py = -tc; py <= tc; ++py) {
				for (int px = -tc; px <= tc; ++px) {
					const float4& pc_color = tex2D(g_img, cx + px, cy + py);
					const float4& pi_color = tex2D(g_img, x + px, y + py);
					const float4& pc_varColor = tex2D(g_imgVar, cx + px, cy + py);
					const float4& pi_varColor = tex2D(g_imgVar, x + px, y + py);

					dist_col += calc_color_distance(pc_color, pi_color, pc_varColor, pi_varColor, epsilon, kc);
				}
			}

			float w = calc_patch_weight(dist_col, tc);

			const float4& i_img = tex2D(g_img, x, y);
			const float4& i_dx = tex2D(g_dx, x, y);
			const float4& i_dy = tex2D(g_dy, x, y);
			outImg += w * i_img;

			outDx += w * i_dx;
			outDy += w * i_dy;

			sumW += w;
		}
	}
	if (sumW < epsilon) {
		_outImg[cIdx] = tex2D(g_img, cx, cy);
		_outDx[cIdx] = tex2D(g_dx, cx, cy);
		_outDy[cIdx] = tex2D(g_dy, cx, cy);
	}
	else {
		_outImg[cIdx] = outImg / sumW;
		_outDx[cIdx] = outDx / sumW;
		_outDy[cIdx] = outDy / sumW;
	}
}

extern "C" void prefilterNlm(float4* _outImg, float4* _outDx, float4* _outDy, const float4* _img, const float4* _imgVar, const float4* _dx, const float4* _dy, int xSize, int ySize) {
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	const int nPix = xSize * ySize;
	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(xSize, blockDim), iDivUp(ySize, blockDim));
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

	cudaArray *g_src_img, *g_src_imgVar, *g_src_dx, *g_src_dy;

	checkCudaErrors(cudaMallocArray(&g_src_img, &channelDesc, xSize, ySize));
	checkCudaErrors(cudaMallocArray(&g_src_imgVar, &channelDesc, xSize, ySize));
	checkCudaErrors(cudaMallocArray(&g_src_dx, &channelDesc, xSize, ySize));
	checkCudaErrors(cudaMallocArray(&g_src_dy, &channelDesc, xSize, ySize));

	checkCudaErrors(cudaMemcpyToArray(g_src_img, 0, 0, _img, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_imgVar, 0, 0, _imgVar, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_dx, 0, 0, _dx, nPix * sizeof(float4), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(g_src_dy, 0, 0, _dy, nPix * sizeof(float4), cudaMemcpyHostToDevice));

	g_img.addressMode[0] = g_img.addressMode[1] = cudaAddressModeMirror;
	g_imgVar.addressMode[0] = g_imgVar.addressMode[1] = cudaAddressModeMirror;
	g_dx.addressMode[0] = g_dx.addressMode[1] = cudaAddressModeMirror;
	g_dy.addressMode[0] = g_dy.addressMode[1] = cudaAddressModeMirror;

	checkCudaErrors(cudaBindTextureToArray(&g_img, g_src_img, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_imgVar, g_src_imgVar, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_dx, g_src_dx, &channelDesc));
	checkCudaErrors(cudaBindTextureToArray(&g_dy, g_src_dy, &channelDesc));

	float4 *d_outImg, *d_outDx, *d_outDy;
	checkCudaErrors(cudaMalloc((void **)&d_outImg, nPix * sizeof(float4)));
	checkCudaErrors(cudaMalloc((void **)&d_outDx, nPix * sizeof(float4)));
	checkCudaErrors(cudaMalloc((void **)&d_outDy, nPix * sizeof(float4)));

	kernel_filter_nlm << <grid, threads >> > (d_outImg, d_outDx, d_outDy, xSize, ySize, PREF_WINDOW_SIZE, PREF_KC, PREF_PATCH_SIZE);

	checkCudaErrors(cudaMemcpy(_outImg, d_outImg, nPix * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_outDx, d_outDx, nPix * sizeof(float4), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_outDy, d_outDy, nPix * sizeof(float4), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_outImg));
	checkCudaErrors(cudaFree(d_outDx));
	checkCudaErrors(cudaFree(d_outDy));

	checkCudaErrors(cudaUnbindTexture(&g_img));
	checkCudaErrors(cudaUnbindTexture(&g_imgVar));
	checkCudaErrors(cudaUnbindTexture(&g_dx));
	checkCudaErrors(cudaUnbindTexture(&g_dy));

	checkCudaErrors(cudaFreeArray(g_src_img));
	checkCudaErrors(cudaFreeArray(g_src_imgVar));
	checkCudaErrors(cudaFreeArray(g_src_dx));
	checkCudaErrors(cudaFreeArray(g_src_dy));

	checkCudaErrors(cudaGetLastError());
}

// ---------------------------------------------------------------------------------
// NLM with symmetric gradient
//static struct {
//	float *_img, *_imgVar, *_dx, *_dy;
//	float *_imgOut, *_dxOut, *_dyOut;
//	
//	// Misc
//	float *_d2avg1, *_d2avg2, *_d2avgS;
//	float *_wgt1, *_wgt2, *_tmp;
//	float *_area;
//} device;

//float sqr(float v) { return v*v; }
//
//#define CLAMP_MIRROR(pos, pos_max) \
//    ((pos) < 0) ? -(pos) : ((pos) >= (pos_max)) ? 2*(pos_max) - (pos) - 2 : (pos)
//#define CALCULATE_INDEX(width, height) \
//    int x = blockIdx.x * blockDim.x + threadIdx.x; \
//    int y = blockIdx.y * blockDim.y + threadIdx.y; \
//    if (x >= (width) || y >= (height)) return; \
//    int index = (y) * (width) + x
//#define CALCULATE_INDICES(width, height, dx, dy) \
//    int x = blockIdx.x * blockDim.x + threadIdx.x; \
//    int y = blockIdx.y * blockDim.y + threadIdx.y; \
//    if (x >= (width) || y >= (height)) return; \
//    int indexC = y * (width) + x; \
//    int x1 = CLAMP_MIRROR(x+(dx), (width)); \
//    int y1 = CLAMP_MIRROR(y+(dy), (height)); \
//    int indexN = x1 + y1 * (width)
//#define CALCULATE_INDICES_SYM(width, height, dx, dy) \
//    int x = blockIdx.x * blockDim.x + threadIdx.x; \
//    int y = blockIdx.y * blockDim.y + threadIdx.y; \
//    if (x >= (width) || y >= (height)) return; \
//    int indexC = y * (width) + x; \
//    int x1 = CLAMP_MIRROR(x+(dx), (width)); \
//    int y1 = CLAMP_MIRROR(y+(dy), (height)); \
//    int indexN1 = x1 + y1 * (width); \
//    int x2 = CLAMP_MIRROR(x-(dx), (width)); \
//    int y2 = CLAMP_MIRROR(y-(dy), (height)); \
//    int indexN2 = x2 + y2 * (width)
//
//__global__ void weights(int width, int height, float * wgt, float const * d2, int dx, int dy)
//{
//	CALCULATE_INDEX(width, height);
//
//	wgt[index] = exp(-max(d2[index], 0.f));
//}
//
//__global__ void weights_sym(int width, int height, float * wgt1, float * wgt2, float const * d2avg1, float const * d2avg2, float const * d2avgS, int dx, int dy)
//{
//	CALCULATE_INDEX(width, height);
//
//	float w1 = exp(-max(0.f, d2avg1[index]));
//	float w2 = exp(-max(0.f, d2avg2[index]));
//	float wS = exp(-max(0.f, d2avgS[index]));
//
//	float wA = w1 + w2;
//	float r = min(1.f, max(0.f, wS / wA - 1));
//	wgt1[index] = r * wS + (1 - r) * w1;
//	wgt2[index] = r * wS + (1 - r) * w2;
//}
//
//__global__ void cumulate(int width, int height, float * target, float const * source)
//{
//	CALCULATE_INDEX(width, height);
//
//	target[index] += source[index];
//}
//
//__global__ void relax(int width, int height, float * tgt, float const * wgt, float const * src, int dx, int dy, int nChannels)
//{
//	CALCULATE_INDICES(width, height, dx, dy);
//
//	for (int c = 0; c < nChannels; c++)
//		tgt[nChannels*indexC + c] += wgt[indexC] * src[nChannels*indexN + c];
//}
//
//__global__ void normalize(int width, int height, float * target, float const * source, float const * area, int nChannels)
//{
//	CALCULATE_INDEX(width, height);
//
//	for (int c = 0; c < nChannels; c++) {
//		target[nChannels*index + c] = source[nChannels*index + c] / area[index];
//	}
//}
//
//
//__global__ void sqr_diff_scale(int width, int height, float * target, float const * source1, float const * source2, float const scale, int nChannels)
//{
//	CALCULATE_INDEX(width, height);
//
//	for (int c = 0; c < nChannels; c++) {
//		float d = source1[nChannels*index + c] - source2[nChannels*index + c];
//		target[nChannels*index + c] = d * d * scale;
//	}
//}
//
//
//__global__ void clamp_min(int width, int height, float * target, float const * source1, float const * source2, int nChannels)
//{
//	CALCULATE_INDEX(width, height);
//
//	for (int c = 0; c < nChannels; c++) {
//		target[nChannels*index + c] = min(target[nChannels*index + c], max(source1[nChannels*index + c], source2[nChannels*index + c]));
//	}
//}
//
//__global__ void clamp_to_zero(int width, int height, float * target, float val, int nChannels)
//{
//	CALCULATE_INDEX(width, height);
//
//	for (int c = 0; c < nChannels; c++) {
//		if (target[nChannels*index + c] < val)
//			target[nChannels*index + c] = 0.f;
//	}
//}