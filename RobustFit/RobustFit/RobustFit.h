#pragma once

#include "RobustFitOptions.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <vector_types.h>
#include <vector>


/* MST (Kruskal's Algorithm) */
// a structure to represent a weighted edge (a gradient) in graph
struct Edge {
	int srcIdx, destIdx;
	float weightVar; // edge weight: estimated gradient error
};

// a structure to represent an undirected graph (vertex: pixel color, edge: gradient) 
struct Graph {
	// V-> Number of vertices(pixel colors), E-> Number of edges(gradients in horizontal & vertical directions)
	int V;
	int E;

	// The graph is represented as an array of edges. 
	// Since the graph is undirected, an edge from src to dest is the same as the one vice versa.
	// Both are counted as 1 edge here.
	std::vector<Edge> edge;
};

// A structure to represent a subset for union and find functions.
struct subset {
	int parent;
	int rank;
};

class RobustFit {
private:
	int m_width, m_height, m_nPix;
	float m_alpha_solver;
	
	// pointers of input images
	const std::vector<float4> *p_throughput;
	const std::vector<float4> *p_dx, *p_dy;	
	const std::vector<float4> *p_varThroughput;
	
	// an edge map indicates whether an edge (a gradient) is included in a gradient subset (0: an edge (a gradient) excluded, 1: an edge (a gradient) included) 
	char2* m_edges; 

	SolverConfig m_config;
public:
	float4* m_optImg;

private:
	// Functions for MST
	static bool compare(const Edge &lhs, const Edge &rhs) {
		return lhs.weightVar < rhs.weightVar;
	}
	int find(std::vector<subset> &subsets, int i);
	void Union(std::vector<subset> &subsets, int x, int y);
	void createGraph(Graph& graph, int nV, int nE);
	void addEdges(Graph &graph, const float4* _errDx, const float4* _errDy, int width, int height);
 
	int runMST(const float4* _errDx, const float4* _errDy, char2* _edges);
	int MSTForward(struct Graph* graph, char2* _edges);

public:
	RobustFit(const std::string preset);
	~RobustFit();
	void clearMemory();
	void allocMemory(int xSize, int ySize, float alpha);
	
	// gradient error estimation using filtering (adaptive NLM by Roussell)
	void estimateGradientErr(float4* _filteredImg, float4* _errDx, float4* _errDy, const float4* img, const float4* varImg, const float4* _dx, const float4* _dy, int xSize, int ySize);
	// finding the optimal gradient subset and solving 
	void calcGraph(char2* _optEdges, const float4 *_tp, const float4 *_dx, const float4 *_dy, const float4 *_errDx, const float4 *_errDy, const float4 *_nlmImg);
};
