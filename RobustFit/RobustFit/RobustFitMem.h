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

#include "RobustFitOptions.h"
#include <vector_types.h>
#include <cuda.h>

struct PixError {
	int idx_grad;
	float error;
};

class RobustFitMem {
public:
	float* _d_key_sort;
	char2* _d_edges_MST;
	char2* _d_edges;
	PixError* _d_pixErr;
	char2* _d_optEdges;

public:
	RobustFitMem() {
		_d_key_sort = NULL;
		_d_edges_MST = NULL;
		_d_edges = NULL;
		_d_pixErr = NULL;
		_d_optEdges = NULL;
	}
	~RobustFitMem() {
		deleteMemory();
	}
	void allocMemory(int width, int height);
	void deleteMemory();
};

class SolverGPUMem {
public:
	float4* _e;
	float4* _e_total;
	float4* _b;
	float4* _r;
	float4* _p;
	float4* _x;
	float*  _w2;
	float4* _AtW2Ap;
	float4* _pq;
	float4* _Ax;

	float4* _alpha;
	float4* _beta;

	float4* _sum_rr_old;
	float4* _sum_rr_new;
	float4* _sum_pq;

	SolverGPUMem(int nPix) {
		checkCudaErrors(cudaMalloc((void **)&_e, 3 * nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_e_total, 3 * nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_b, 3 * nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_r, nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_p, nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_x, nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_w2, 3 * nPix * sizeof(float)));
		checkCudaErrors(cudaMalloc((void **)&_AtW2Ap, nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_pq, nPix * sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_Ax, 3 * nPix * sizeof(float4)));

		checkCudaErrors(cudaMalloc((void **)&_alpha, sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_beta, sizeof(float4)));

		checkCudaErrors(cudaMalloc((void **)&_sum_rr_old, sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_sum_rr_new, sizeof(float4)));
		checkCudaErrors(cudaMalloc((void **)&_sum_pq, sizeof(float4)));

		checkCudaErrors(cudaGetLastError());
	}
	~SolverGPUMem() {
		checkCudaErrors(cudaFree(_e));
		checkCudaErrors(cudaFree(_e_total));
		checkCudaErrors(cudaFree(_b));
		checkCudaErrors(cudaFree(_r));
		checkCudaErrors(cudaFree(_p));
		checkCudaErrors(cudaFree(_x));
		checkCudaErrors(cudaFree(_w2));	
		checkCudaErrors(cudaFree(_AtW2Ap));
		checkCudaErrors(cudaFree(_pq));
		checkCudaErrors(cudaFree(_Ax));

		checkCudaErrors(cudaFree(_alpha));
		checkCudaErrors(cudaFree(_beta));

		checkCudaErrors(cudaFree(_sum_rr_old));
		checkCudaErrors(cudaFree(_sum_rr_new));
		checkCudaErrors(cudaFree(_sum_pq));

		checkCudaErrors(cudaGetLastError());
	}
};