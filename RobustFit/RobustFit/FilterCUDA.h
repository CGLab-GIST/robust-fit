#pragma once

#include <vector_types.h>

extern "C" void prefilterNlm(float4* _outImg, float4* _outDx, float4* _outDy, const float4* _img, const float4* _varImg, const float4* _dx, const float4* _dy, int xSize, int ySize);