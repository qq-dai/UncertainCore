#pragma once
#include "graph.h"
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <functional>
#include <unistd.h>
#include <sys/stat.h>
#include <gsl/gsl_sf_gamma.h>
#include <gmp.h>

using namespace std;

#define vec_i vector<int>
#define vec_b vector<bool>
#define vec_d vector<Double>

class Uncertain_Core : public Graph
{
public:
	Uncertain_Core();
	~Uncertain_Core();

	void set_threads(int th);
	void set_eta(Double eta);
	void get_sorted_adj(int alg);

	// original dp KDD14 and ICDE19
	void truncated_uncertain_core_decomposition(vec_i &coreness, Double eta);
	void truncated_uncertain_core_decomposition_muti_precision(vec_i &coreness, Double eta);

	//lower bound + upper bound to improve
	int upper_bound(vec_i &UB);
	//14 parallel k-core
	int upper_bound_parallel(vec_i &UB);
	//17 parallel k-core
	int upper_bound_parallel_new(vec_i &UB);

	//SemiCore ICDE2016
	void semicore(vec_i &UB);

	void lower_bound(vec_i &LB);
	//edge-parallel strategy
	int lower_bound_parallel(vec_i &LB, vec_i &UB);

	//KDD14
	void lower_bound_of_beta_fun(vec_i &LB);
	//edge-parallel strategy
	void lower_bound_of_beta_fun_parallel(vec_i &LB);

	//KDD14 with recomputing eta-degree
	void Basic_precise_core_decomposition(vec_i &coreness, Double eta);

	//on demaond algorithm
	void BottomUp_core_decomposition(vec_i &coreness, Double eta, int prune);
	//lazy update
	void ImpBottomUp_core_decomposition_parallel(vec_i &coreness, Double eta, int prune);

	//Basic top-down algorithm
	void Basic_topdown_core_decomposition_parallel(vec_i &coreness, Double eta);

	//optimized top-down algorithm with the binary/isometrics partition
	void Optimal_topdown_core_decomposition_parallel(vec_i &coreness, Double eta);

private:
	Double eta;
	int threads, buf_th_size;

	pair<Double, int> *adj_nbrs, **Adj;
	vec_i Del, is_Del;
	int *leftc, *leftn;

	int eta_degree(Double *pro_new, Double *pro_old, int v);
	int eta_degree(Double *pro_new, Double *pro_old, int v, int UB_v);
	int eta_degree_mpf(mpf_t *pro_new, mpf_t *pro_old, int v, int UB_v);

	int recompute_eta_degree(vec_b &visited, vec_i &pos, Double *pro_new, Double *pro_old, int v, int ub);
	int recompute_eta_degree(vec_i &visited, Double *pos, Double *pro_new, Double *pro_old, int v, int ub);
	int recompute_eta_degree_mpf(vec_i &visited, int *pos, mpf_t *pro_new, mpf_t *pro_old, int v, int ub);

	//top-down
	int update_eta_degree(vec_i &visited, Double *pos, Double *pro_new, Double *pro_cur, 
		int cur_eta_deg, int cur_core, int v, int *UB);

	int dp_eta_deg(vec_i &visited, Double *pos, Double *pro_new, Double *pro_old, Double *res, int v, int ub);
	int apped_eta_deg(Double *apped, int app_size, Double *pro_new, Double *pro_old, Double *res, int ub);
	int reset_cur_deg(vec_i &removed, vec_i &cur_deg, int v, int max_lb, int max_ub, vec_i &bin);
};
