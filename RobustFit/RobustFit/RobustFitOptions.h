#pragma once

#define RECONST_GPT        1
#define RECONST_LTS        2
#define RECONST_OPTIONS    RECONST_LTS

#define ITER_START_PERCENT 50	// initial gradient subset for LTS 
#define ITER_AUGMENTATION  11	// setting the augmentation rate of subset sizes (11 times: from 50% to 100%, augments by 5% rate)
#define DEBUG_SOLVER			// Timing functions

// NLM parameters
#define PREF_WINDOW_SIZE   9
#define PREF_PATCH_SIZE    3 
#define PREF_KC            0.45 

#include <vector_types.h>
#include <stdlib.h>
#include <memory.h>
#include <random>

struct SolverConfig {
	int irls_max_iter;
	float irls_reg_init;
	float irls_reg_iter;
	int cg_max_iter;
};