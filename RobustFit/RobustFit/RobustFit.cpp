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

#include "RobustFit.h"
#include "helper_math.h"
#include <random>
#include <omp.h>
#include <iostream>
#include <iterator>
#include <time.h>
#include <set>

#include "RobustFitCUDA.h"
#include "FilterCUDA.h"

RobustFit::RobustFit(const std::string preset) {
	m_optImg = NULL;

	if (preset == "L1") {
		m_config.irls_max_iter = 20;
		m_config.irls_reg_init = 0.05f;
		m_config.irls_reg_iter = 0.5f;
		m_config.cg_max_iter = 50;

	}
	else if (preset == "L2") {
		m_config.irls_max_iter = 1;
		m_config.irls_reg_init = 0.0f;
		m_config.irls_reg_iter = 0.0f;
		m_config.cg_max_iter = 50;
	}
	
	m_edges = NULL;
}

void RobustFit::allocMemory(int xSize, int ySize, float alpha) {
	m_alpha_solver = alpha;
	m_width = xSize;
	m_height = ySize;
	m_nPix = xSize * ySize;

	clearMemory();

	m_optImg = new float4[m_nPix];
	m_edges = new char2[m_nPix];
}

void RobustFit::clearMemory() {
	if (m_optImg) delete[] m_optImg;
}

RobustFit::~RobustFit() {
	clearMemory();
}

/// Funcionts for MST: create a graph with V vertices(colors) and E edges(gradients)
void RobustFit::createGraph(Graph& graph, int nV, int nE) {
	graph.V = nV; // the number of vertices (colors)
	graph.E = nE; // the number of edges (gradients)
	graph.edge.resize(nE);
}

/// Funcionts for MST: add edges (gradients) and set weights (gradients with gradient error estimation) 
void RobustFit::addEdges(Graph &graph, const float4* _errDx, const float4* _errDy, int width, int height) {
	int edgeIdx = 0;

	//#pragma omp parallel for schedule(guided, 4)
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (edgeIdx < graph.E)
			{
				int src = y * width + x;

				if (x < width - 1) {
					int destRight = src + 1;
					float weightDx = (fabs(_errDx[src].x) + fabs(_errDx[src].y) + fabs(_errDx[src].z)) / 3.f;

					graph.edge[edgeIdx].srcIdx = src;
					graph.edge[edgeIdx].destIdx = destRight;
					graph.edge[edgeIdx].weightVar = weightDx;
					edgeIdx++;
				}

				if (y < height - 1)	{
					int destBottom = src + width;
					float weightDy = (fabs(_errDy[src].x) + fabs(_errDy[src].y) + fabs(_errDy[src].z)) / 3.f;
					graph.edge[edgeIdx].srcIdx = src;
					graph.edge[edgeIdx].destIdx = destBottom;
					graph.edge[edgeIdx].weightVar = weightDy;
					edgeIdx++;
				}
			}
		}
	}

}

/// An utility function to find set of an element i
/// (uses path compression technique)
int RobustFit::find(std::vector<subset> &subsets, int i)
{
	// find root and make root as parent of i (path compression)
	if (subsets[i].parent != i)
		subsets[i].parent = find(subsets, subsets[i].parent);

	return subsets[i].parent;
}

/// A function that does union of two sets of x and y
/// (uses Union by rank)
void RobustFit::Union(std::vector<subset> &subsets, int x, int y)
{
	int xroot = find(subsets, x);
	int yroot = find(subsets, y);

	// Attach smaller rank tree under root of high rank tree
	// (Union by Rank)
	if (subsets[xroot].rank < subsets[yroot].rank)
		subsets[xroot].parent = yroot;
	else if (subsets[xroot].rank > subsets[yroot].rank)
		subsets[yroot].parent = xroot;


	// If ranks are same, then make one as root and increment
	// its rank by one
	else
	{
		subsets[yroot].parent = xroot;
		subsets[xroot].rank++;
	}
}

/// MST computation
int RobustFit::MSTForward(struct Graph* graph, char2* _edges) {
	int V = graph->V;
	int E = graph->E;

	std::vector<Edge> result(V - 1); // This will store the resultant MST

	int e = 0; // An index variable, used for result[]
	int i = 0; // An index variable, used for sorted edges
	int j = 0; // An index variable, iterated edges(whether it's in MST or not)

	std::stable_sort(graph->edge.begin(), graph->edge.end(), compare);

	// Allocate memory for creating V ssubsets
	std::vector<subset> subsets(V);

	// Create V subsets with single elements
	for (int v = 0; v < V; ++v)
	{
		if (subsets.size() <= v) {
			printf("ERR1\n\n");
		}
		subsets[v].parent = v;
		subsets[v].rank = 0;
	}

	// Number of edges to be taken is equal to V-1
	while (e < V - 1)
	{
		// Pick the smallest edge. And increment the index
		// for next iteration
		if (graph->edge.size() <= i) {
			printf("ERR2\n\n");
		}
		struct Edge next_edge = graph->edge[i++];

		int x = find(subsets, next_edge.srcIdx);
		int y = find(subsets, next_edge.destIdx);

		// If including this edge does't cause cycle, include it
		// in result and increment the index of result for next edge
		if (x != y)	{
			if (result.size() <= e) {
				printf("ERR3\n\n");
			}
			result[e++] = next_edge;
			Union(subsets, x, y);
		}
	}
	printf("Number of Edges in MST = %d, nPix = %d\n", e, m_nPix);

	memset(_edges, 0, sizeof(char2) * m_nPix);
	for (int k = 0; k < result.size(); k++)
	{
		int index = result[k].srcIdx;
		
		if (result[k].destIdx - index == 1) // dX edge selected		
			_edges[index].x = 1;
		else if (result[k].destIdx - index == m_width)  // dY edge selected		
			_edges[index].y = 1;
		else 
			printf("ERR in MST: Cannot reach to here!\n");
	}
	return e;
}

/// call MST precess functions 
int RobustFit::runMST(const float4* _errDx, const float4* _errDy, char2* _edges) {
	int V = m_nPix; // Number of vertices in graph
	int E = 2 * m_nPix - m_width - m_height;

	// Create a graph with edges															
	Graph graph;	
	createGraph(graph, V, E);

	addEdges(graph, _errDx, _errDy, m_width, m_height);

	// MST
	int nEdges = MSTForward(&graph, _edges);
	return nEdges;
}

void RobustFit::calcGraph(char2* _optEdges, const float4 *_tp, const float4 *_dx, const float4 *_dy, 
	const float4 *_errDx, const float4 *_errDy, const float4 *_nlmImg) {
#ifdef DEBUG_SOLVER  
	clock_t sTime, eTime;
	double elapsed_secs_total = 0.0;
	sTime = clock();
#endif
	// Step 1. Run MST
	int nEdges = runMST(_errDx, _errDy, m_edges);

#ifdef DEBUG_SOLVER  
	eTime = clock();
	printf("MST Time = %.4f secs | Initial edge = %d\n", (double)(eTime - sTime) / CLOCKS_PER_SEC, nEdges);
	elapsed_secs_total += (double)(eTime - sTime);
	sTime = clock();
#endif
	// Step 2. Run RLS based edge (gradient subset) augmenting	
	robust_fit_cuda(m_config, m_alpha_solver, m_edges, _tp, _dx, _dy, _errDx, _errDy, _nlmImg, m_width, m_height, nEdges, m_optImg);

#ifdef DEBUG_SOLVER  
	eTime = clock();
	printf("Least Trimmed Squares = %.4f secs\n", (double)(eTime - sTime) / CLOCKS_PER_SEC);
	elapsed_secs_total += (double)(eTime - sTime);
#endif
}
// Estimation for gradient errors using denoising filter
void RobustFit::estimateGradientErr(float4* _fltImg, float4* _errDx, float4* _errDy, const float4* img, const float4* varImg, const float4* _dx, const float4* _dy, int xSize, int ySize) {
#ifdef DEBUG_SOLVER  
	clock_t sTime, eTime;
	double elapsed_secs_total = 0.0;
	sTime = clock();
#endif
	int nPix = xSize * ySize;

	float4* _fltDx = new float4[nPix];
	float4* _fltDy = new float4[nPix];

	// filtering by nlm (Rousselle et al.) -> used for ground truth estimation
	prefilterNlm(_fltImg, _fltDx, _fltDy, img, varImg, _dx,  _dy, xSize, ySize);
	float epsilon = 0.01;

#pragma omp parallel for
	for (int y = 0; y < ySize; ++y) {
		for (int x = 0; x < xSize; ++x) {
			int i = y * xSize + x;

			float4 eDx, eDy;

			eDx.x = _dx[i].x - _fltDx[i].x;
			eDx.y = _dx[i].y - _fltDx[i].y;
			eDx.z = _dx[i].z - _fltDx[i].z;

			eDy.x = _dy[i].x - _fltDy[i].x;
			eDy.y = _dy[i].y - _fltDy[i].y;
			eDy.z = _dy[i].z - _fltDy[i].z;

			_errDx[i] = make_float4(eDx.x * eDx.x, eDx.y * eDx.y, eDx.z * eDx.z, 0.f);
			_errDy[i] = make_float4(eDy.x * eDy.x, eDy.y * eDy.y, eDy.z * eDy.z, 0.f);
		}
	}

	delete[] _fltDx;
	delete[] _fltDy;
#ifdef DEBUG_SOLVER  
	eTime = clock();
	elapsed_secs_total = (double)(eTime - sTime);
	printf("NLM filtering Time (Total) = %.4f secs\n", elapsed_secs_total / CLOCKS_PER_SEC);
#endif
}