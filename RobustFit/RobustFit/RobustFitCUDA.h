#pragma once

#include "RobustFitOptions.h"
#include <vector_types.h>

extern "C" void robust_fit_cuda(SolverConfig solver_config, float initAlpha, char2* _edges,
	const float4* _tp, const float4* _dx, const float4* _dy, const float4* _varDx, const float4* _varDy,
	const float4* _nlmImg, int xSize, int ySize, const int nInitEdges,
	float4* _optImg);