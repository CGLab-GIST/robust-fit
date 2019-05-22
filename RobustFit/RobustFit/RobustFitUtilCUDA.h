#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "helper_cuda.h"
#include "helper_math.h"

// Utility functions
static __host__ __device__ float4 max(const float4& a, const float4& b) {
	float4 c;
	c.x = max(a.x, b.x);
	c.y = max(a.y, b.y);
	c.z = max(a.z, b.z);
	c.w = max(a.w, b.w);
	return c;
}

static __host__ __device__ float4 min(const float4& a, const float4& b) {
	float4 c;
	c.x = min(a.x, b.x);
	c.y = min(a.y, b.y);
	c.z = min(a.z, b.z);
	c.w = min(a.w, b.w);
	return c;
}

static __host__ __device__ float4 max(const float4& a, const float& b) {
	float4 c;
	c.x = max(a.x, b);
	c.y = max(a.y, b);
	c.z = max(a.z, b);
	c.w = max(a.w, b);
	return c;
}