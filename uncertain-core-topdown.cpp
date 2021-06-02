#include "uncertain-core.h"

int Uncertain_Core::dp_eta_deg(vec_i &is_removed, Double *pos, Double *pro_new, Double *pro_old, Double *res, int v, int ub)
{
	int d = deg[v];
	if (d <= 0 || ub <= 0) return 0;
	int u, pos_size = 0;
	Double p;
	for (int i = 0; i < d; ++i)
	{
		u = adj[v][i].u;
		if (!is_removed[u])
		{
			p = adj[v][i].p;
			pos[pos_size++] = p;
		}
	}
	if (pos_size <= 0) return 0;
	Double *pro_temp, pro = 1;
	pro_new[0] = pro_old[0] = 1;
	for (int i = 1; i <= pos_size; ++i){
		p = pos[i-1];
		pro_new[0] = pro_old[0] * (1-p);
		pro = 1; pro -= pro_new[0];
		for (int j = 1; j < i && j < ub; ++j) {
			pro_new[j] = pro_old[j - 1] * p + pro_old[j] * (1 - p);
			pro -= pro_new[j];
		}
		if (i < ub) pro_new[i] = pro_old[i-1] * p;
		else if (pro - 1e-15 > eta){
			return ub;
		}
		pro_temp = pro_new;
		pro_new = pro_old;
		pro_old = pro_temp;
	}
	pro_temp = pro_new;
	pro_new = pro_old;
	pro_old = pro_temp;
	pro = 1;
	if (res != NULL) {
		d = ub;
		for (int i = 0; i < ub; ++i) {
			pro -= pro_new[i];
			res[i] = pro_new[i];
			if ( d == ub && pro + 1e-15 < eta)
				d = i;
		}
		return d;
	}
	for (int i = 0; i < ub; ++i) {
		pro -= pro_new[i];
		if (pro + 1e-15 < eta) {
			return i;
		}
	}
	return ub;
}

int Uncertain_Core::apped_eta_deg(Double *apped, int app_size, Double *pro_new, Double *pro_old, Double *res, int ub)
{
	Double *pro_temp, p, pro = 1;
	if (res == NULL) {
		printf("func: apped_eta_deg, res=NULL\n");
		abort();
	}
	for (int i = 0; i < app_size; ++i) {
		p = apped[i];
		pro_new[0] = pro_old[0] * (1-p);
		pro = 1; pro -= pro_new[0];
		for (int j = 1; j < ub; ++j) {
			pro_new[j] = pro_old[j - 1] * p + pro_old[j] * (1 - p);
			pro -= pro_new[j];
		}
		pro_temp = pro_new;
		pro_new = pro_old;
		pro_old = pro_temp;
	}
	pro_temp = pro_new;
	pro_new = pro_old;
	pro_old = pro_temp;
	pro = 1;
	if (res != NULL) {
		int d = ub;
		for (int i = 0; i < ub; ++i) {
			pro -= pro_new[i];
			res[i] = pro_new[i];
			if ( d == ub && pro + 1e-15 < eta)
				d = i;
		}
		return d;
	}
	for (int i = 0; i < ub; ++i) {
		pro -= pro_new[i];
		if (pro + 1e-15 < eta)
			return i;
	}
	return ub;
}

int Uncertain_Core::reset_cur_deg(vec_i &removed, vec_i &cur_deg, int v, int max_lb, int max_ub, vec_i &bin)
{
	int cudv = cur_deg[v];
	if (cudv <= max_lb) return cudv;
	bin.clear(); bin.resize(max_ub+1, 0);
	int d = deg[v];
	int degn = 0;
	for (int j = 0; j < d; ++j) {
		int u = adj[v][j].u;
		if (!removed[v]){
			int cudu = cur_deg[u];
			bin[min(max_ub, cudu)]++;
			adj[v][degn].u = u;
			adj[v][degn++].p = adj[v][j].p;
		}
	}
	deg[v] = degn;
	//return min(cudv, max_ub);
	d = 0;
	for (int j = max_ub; j >= max_lb; --j){
		d += bin[j];
		if (d >= j)
			return min(cudv, j);
	}
	return max_lb;
}

// without optimizations
void Uncertain_Core::Basic_topdown_core_decomposition_parallel(vec_i &coreness, Double eta)
{
	double pt, start_tm = omp_get_wtime();
	this->eta = eta;
	coreness.clear(); coreness.resize(n, 0);
	vec_i LB, UB, UB_new;
	int max_ub = 0, max_lb = 0, max_core = 0;
	max_ub = upper_bound_parallel_new(UB);
	max_lb = lower_bound_parallel(LB, UB);
	printf("max_lb = %d, max_ub = %d \n", max_lb, max_ub);
	pt = omp_get_wtime();

	Double **pos_threads, **pre_threads, **now_threads;
	pos_threads = new Double*[threads];
	pre_threads = new Double*[threads];
	now_threads = new Double*[threads];
	for (int i = 0; i < threads; ++i) {
		pos_threads[i] = new Double[n];
		pre_threads[i] = new Double[max_degree+1];
		now_threads[i] = new Double[max_degree+1];
	}
	Double *pos, *pro_new, *pro_old, *probability, **eta_pro;
	pos = pos_threads[0];
	pro_new = now_threads[0];
	pro_old = pre_threads[0];
	
	probability = new Double[m];
	eta_pro = new Double*[n];

	vec_i visited(n, 0), removed(n, 1), eta_Deg(n, 0);
	vector < vec_i > vertices(max_ub+1);
	for (int i = 0; i < n; ++i) {
		int ub = UB[i];
		vertices[ub].push_back(i);
	}
	eta_pro[0] = probability;
	for (int i = 0; i < n - 1; ++i) {
		eta_pro[i+1] = eta_pro[i] + deg[i];
	}

	int del_size = 0, isdel_size = 0;
	int cnt = 0;
	int init_minc = max_lb, init_maxc = max_ub;
	int mid_core = 0;
	int remove_size = 0;
	int cur_core = max_ub+1;
	int left_size = 0;
	vec_i Removend(n), left_nodes(n), cur_deg(n, 0), cur_deg_new(n, 0);
	
	#pragma omp parallel
	{
		int x;
		#pragma omp for schedule(dynamic)
		for (int k = max_lb; k <= max_ub; ++k) {
			size_t size = vertices[k].size();
			#pragma omp atomic capture
			x = del_size += size;
			x -= size;
			for (int i = 0; i < size; ++i) {
				int v = vertices[k][i];
				removed[v] = 0;
				Del[x] = v;
				left_nodes[x++] = v;
			}
		}
		left_size = del_size;
		#pragma omp barrier
		int pid = omp_get_thread_num();
		Double *pos_thread = pos_threads[pid];
		Double *pre_thread = pre_threads[pid];
		Double *now_thread = now_threads[pid];
		int Buf[128];
		int buf_cap = 128, buf_size = 0; 
		#pragma omp for schedule (dynamic, 4)
		for (int i = 0; i < del_size; ++i) {
			int v = Del[i];
			int d = dp_eta_deg(removed, pos_thread, now_thread, pre_thread, eta_pro[v], v, UB[v]);
			eta_Deg[v] = d;
			cur_deg[v] = d;
			if (d < max_lb) {
				Buf[buf_size++] = v;
				if (buf_size == buf_cap){
					buf_size = 0;
					#pragma omp atomic capture
					x = isdel_size += buf_cap;
					x -= buf_cap;
					for (int h = 0; h < buf_cap; ++h)
						is_Del[x++] = Buf[h];
				}
			}
		}
		#pragma omp atomic capture
		x = isdel_size += buf_size;
		x -= buf_size;
		for (int h = 0; h < buf_size; ++h)
			is_Del[x++] = Buf[h];
		buf_size = 0;
	}
	cnt =  del_size - isdel_size;

	//Detecting (maxlb, eta)-core
	while (isdel_size > 0) {
		remove_size = 0;
		#pragma omp parallel if (isdel_size > 3)
		{
			int pid = omp_get_thread_num();
			Double *pos_thread = pos_threads[pid];
			Double *pre_thread = pre_threads[pid];
			Double *now_thread = now_threads[pid];
			int Buf[128];
			int buf_cap = 128, buf_size = 0; 

			int x;
			#pragma omp for schedule(dynamic)
			for (int i = 0; i < isdel_size; ++i) {
				int v = is_Del[i];
				int d = deg[v];
				removed[v] = 1;
				for (int j = 0; j < d; ++j) {
					int u = adj[v][j].u;
					if (!removed[u] && cur_deg[u] >= max_lb)
						if (!visited[u] ){
							//if (visited[u] > 0) continue;
							#pragma omp atomic capture
							x = visited[u]++;
							if (x == 0) {
								// #pragma omp atomic capture
								// x = remove_size++;
								// Removend[x] = u;
								// continue;
								Buf[buf_size++] = u;
								if (buf_size == buf_cap){
									buf_size = 0;
									#pragma omp atomic capture
									x = remove_size += buf_cap;
									x -= buf_cap;
									for (int h = 0; h < buf_cap; ++h)
										Removend[x++] = Buf[h];
								}
							}
						}
				}
			}
			isdel_size = 0;
			#pragma omp atomic capture
			x = remove_size += buf_size;
			x -= buf_size;
			for (int h = 0; h < buf_size; ++h)
				Removend[x++] = Buf[h];
			buf_size = 0;
			#pragma omp barrier
	
			#pragma omp for schedule (dynamic)
			for (int i = 0; i < remove_size; ++i) {
				int d = 0, v = Removend[i];
				visited[v] = 0;
				if (cnt < max_lb) continue;
				d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, UB[v]);
				cur_deg[v] = d;
				if (d < max_lb){
					removed[v] = 1;
					Buf[buf_size++] = v;
					if (buf_size == buf_cap){
						buf_size = 0;
						#pragma omp atomic capture
						x = isdel_size += buf_cap;
						if (cnt < max_lb + x) cnt = 0;
						x -= buf_cap;
						for (int h = 0; h < buf_cap; ++h)
							is_Del[x++] = Buf[h];
					}
					// #pragma omp atomic capture
					// x = isdel_size++;
					// is_Del[x] = v;
					// if (cnt < max_lb + x)
					// 	cnt = 0;
				}
			}
			#pragma omp atomic capture
			x = isdel_size += buf_size;
			x -= buf_size;
			for (int h = 0; h < buf_size; ++h)
				is_Del[x++] = Buf[h];
			buf_size = 0;
		}
		cnt -= isdel_size;
		if (cnt < max_lb) isdel_size = 0;
	}
	printf("Time: %f s, lb_core=%d, cnt=%d\n", omp_get_wtime() - pt, max_lb, cnt);

	vec_i deg_lb(n, 0);
	vector <vec_i>  sortednodes(max_ub+1);
	int xxx = 0, del_size_new = 0;
	for (int i = 0; i < del_size; ++i) {
		int v = Del[i];
		int d = cur_deg[v];
		if (d < max_lb) continue;
		sortednodes[d].push_back(v);
		deg_lb[v] = d;
		cur_deg_new[v] = d;
		Del[del_size_new++] = v;
		if (d == max_lb)
			coreness[v] = max_lb;
	}
	del_size = del_size_new;

	//Detecting the maximum eta-core with the binary search 
	init_minc = max_lb+1; init_maxc = max_ub;
	while (init_minc <= init_maxc) {
		int cnt_new = cnt;
		mid_core = (init_minc + init_maxc) / 2;
		isdel_size = 0;
		for (int i = 0; i < del_size; ++i) {
			int v = Del[i];
			if (cur_deg_new[v] < mid_core || UB[v] < mid_core) {
				removed[v] = 1;
				is_Del[isdel_size++] = v;
			}
		}
		cnt_new = del_size - isdel_size;
		while (isdel_size > 0 && cnt_new >= mid_core) {
			remove_size = 0;
			#pragma omp parallel if (isdel_size > 3)
			{
				int pid = omp_get_thread_num();
				Double *pos_thread = pos_threads[pid];
				Double *pre_thread = pre_threads[pid];
				Double *now_thread = now_threads[pid];
				int Buf[128];
				int buf_cap = 128, buf_size = 0; 
				int x;

				#pragma omp for schedule(dynamic)
				for (int i = 0; i < isdel_size; ++i) {
					int v = is_Del[i];
					int d = deg[v];
					removed[v] = 1;
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						if (!removed[u] && !visited[u]){
							if (visited[u] > 0) continue;
								#pragma omp atomic capture
								x = visited[u]++;
								if (!x) {
								// #pragma omp atomic capture
								// x = remove_size++;
								// Removend[x] = u;

								Buf[buf_size++] = u;
								if (buf_size == buf_cap){
									buf_size = 0;
									#pragma omp atomic capture
									x = remove_size += buf_cap;
									x -= buf_cap;
									for (int h = 0; h < buf_cap; ++h)
										Removend[x++] = Buf[h];
								}
							}
						}
					}
				}
			
				isdel_size = 0;
				#pragma omp atomic capture
				x = remove_size += buf_size;
				x -= buf_size;
				for (int h = 0; h < buf_size; ++h)
					Removend[x++] = Buf[h];
				#pragma omp barrier
				buf_size = 0;
				#pragma omp for schedule(dynamic)
				for (int i = 0; i < remove_size; ++i) {
					int d, v = Removend[i];
					visited[v] = 0;
					if (cnt_new < mid_core) continue;
					d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, UB[v]);
					cur_deg_new[v] = d;
					if (d < mid_core){
						removed[v] = 1;
						Buf[buf_size++] = v;
						if (buf_size == buf_cap){
							buf_size = 0;
							#pragma omp atomic capture
							x = isdel_size += buf_cap;
							if (cnt_new < mid_core + x) cnt_new = 0;
							x -= buf_cap;
							for (int h = 0; h < buf_cap; ++h)
								is_Del[x++] = Buf[h];
						}
					}
				}
				#pragma omp atomic capture
				x = isdel_size += buf_size;
				x -= buf_size;
				for (int h = 0; h < buf_size; ++h)
					is_Del[x++] = Buf[h];
			}
			cnt_new -= isdel_size;
			if (cnt_new < mid_core) isdel_size = 0;
		}
		if (cnt_new < mid_core) {
			init_maxc = mid_core - 1;
			max_core = init_maxc;
			for (int i = 0; i < del_size; ++i) {
				int v  = Del[i];
				int d = cur_deg[v];
				cur_deg_new[v] = d;
				removed[v] = 0;
			}
		}
		else {
			init_minc = mid_core+1;
			cnt = cnt_new;
			max_core = mid_core; del_size_new = 0;
			xxx = 0;
			for (int i = 0; i < del_size; ++i) {
				int v  = Del[i];
				int d = cur_deg_new[v];
				cur_deg[v] = d;
				if (d < mid_core) continue;
				Del[del_size_new++] = v;
				//if (d == mid_core) coreness[v] = mid_core;
				coreness[v] = mid_core;
			}
			del_size = del_size_new;
		}
	}
	for (int i = 0; i < del_size; ++i) {
		int v  = Del[i];
		coreness[v] = max_core;
	}

	printf("Time: %f s, max_core=%d, cnt=%d\n", omp_get_wtime() - pt, max_core, cnt);
	
	// Computing eta-cores in the interval [maxlb, max_core-1]
	cur_core = max_core - 1;
	del_size = 0;
	int cnt_core = 0, cnt_cores= 0;
	while (cur_core >= max_lb) {
		isdel_size = 0; del_size = 0;
		int cnt_size = 0;
		#pragma omp parallel 
		{
			int Buf[128], x;
			int buf_cap = 128, buf_size = 0, size_new = 0; 
			#pragma omp for reduction(+:cnt_core, cnt_size)
			for (int i = cur_core; i <= max_ub; i ++){
				size_t size = sortednodes[i].size();
				size_new = 0;
				for (int j = 0; j < size; ++j) {
					int v = sortednodes[i][j];
					removed[v] = 0;
					if (coreness[v] >= cur_core) {
						cnt_core++;
						continue;
					}
					cnt_size++;
					sortednodes[i][size_new++] = v;
					// #pragma omp atomic capture
					// x = del_size++;
					// Del[x] = v;
					Buf[buf_size++] = v;
					if (buf_size == buf_cap){
						buf_size = 0;
						#pragma omp atomic capture
						x = del_size += buf_cap;
						x -= buf_cap;
						for (int h = 0; h < buf_cap; ++h)
							Del[x++] = Buf[h];
					}
					int d = deg[v];
					bool flag = false;
					for (int l = 0; l < d; ++l) {
						int u = adj[v][l].u;
						int d = deg_lb[u];
						if (d >= max_lb && d < cur_core)
							flag = true;
					}
					if (flag) {
						#pragma omp atomic capture
						x = isdel_size++;
						is_Del[x] = v;
					}
				}
				sortednodes[i].resize(size_new);
			}
			#pragma omp atomic capture
			x = del_size += buf_size;
			x -= buf_size;
			for (int h = 0; h < buf_size; ++h)
				Del[x++] = Buf[h];
		}
		int cnt_new = 0;
		while(isdel_size > 0) {
			remove_size = 0;

			#pragma omp parallel if (isdel_size > 3)
			{
				int pid = omp_get_thread_num();
				Double *pos_thread = pos_threads[pid];
				Double *pre_thread = pre_threads[pid];
				Double *now_thread = now_threads[pid];
				int Buf[128];
				int buf_cap = 128, buf_size = 0; 
				int x;
				#pragma omp for schedule(dynamic) reduction(+:cnt_new)
				for (int i = 0; i < isdel_size; ++i) {
					int d, v = is_Del[i];
					visited[v] = 0;
					d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, cur_core);
					if (d < cur_core) {
						removed[v] = 1;
						// #pragma omp atomic capture
						// x = remove_size++;
						// Removend[x] = v;
						cnt_new ++;
						Buf[buf_size++] = v;
						if (buf_size == buf_cap){
							buf_size = 0;
							#pragma omp atomic capture
							x = remove_size += buf_cap;
							x -= buf_cap;
							for (int h = 0; h < buf_cap; ++h)
								Removend[x++] = Buf[h];
						}
					}
				}
				#pragma omp atomic capture
				x = remove_size += buf_size;
				x -= buf_size;
				for (int h = 0; h < buf_size; ++h)
					Removend[x++] = Buf[h];
				buf_size = 0;
				isdel_size = 0;
				#pragma omp barrier

				#pragma omp for schedule(dynamic)
				for (int i = 0; i < remove_size; ++i) {
					int v = Removend[i];
					int d = deg[v];
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						int core = coreness[u];
						if (!removed[u] && core < cur_core)
						if (!visited[u]){
							#pragma omp atomic capture
							x = visited[u]++;
							if (x == 0) {
								// #pragma omp atomic capture
								// x = isdel_size++;
								// is_Del[x] = u;
								Buf[buf_size++] = u;
								if (buf_size == buf_cap){
									buf_size = 0;
									#pragma omp atomic capture
									x = isdel_size += buf_cap;
									x -= buf_cap;
									for (int h = 0; h < buf_cap; ++h)
										is_Del[x++] = Buf[h];
								}
							}
						}
					}
				}
				#pragma omp atomic capture
				x = isdel_size += buf_size;
				x -= buf_size;
				for (int h = 0; h < buf_size; ++h)
					is_Del[x++] = Buf[h];
			}
		}
		#pragma omp parallel for
		for (int i = 0; i < del_size; ++i){
			int v = Del[i];
			if (!removed[v])
				coreness[v] = cur_core;
		}
		cur_core--;
	}

	//Computing eta-cores in the interval (0, maxlb - 1]
	int left_size_new = 0;
	int recompute_size = 0;
	int overall_size = left_size;
	cnt_cores = 0;
	for (int i = 0; i < left_size; ++i) {
		int v = left_nodes[i];
		removed[v] = 0;
		if (coreness[v] >= cur_core)
			cnt_cores ++;
		else
			left_nodes[left_size_new++] = v;
	}
	left_size = left_size_new;

	Double **pro_update = new Double*[threads];
	for (int i = 0; i < threads; ++i)
		pro_update[i] = new Double[max_degree+1];
	for (int k = cur_core; k > 0; k--) {
		size_t size = vertices[k].size();
		int cnt_new = 0, cur_size = left_size;
		overall_size += size;
		for (int i = 0; i < size; ++i) {
			int v = vertices[k][i];
			removed[v] = 0;
			left_nodes[left_size++] = v;
		}
		isdel_size = 0; recompute_size = 0;
		#pragma omp parallel
		{
			int pid = omp_get_thread_num();
			Double *pos_thread = pos_threads[pid];
			Double *pre_thread = pre_threads[pid];
			Double *now_thread = now_threads[pid];
			int Buf[128];
			int buf_cap = 128, buf_size = 0; 
			int x;
			#pragma omp for schedule(dynamic) reduction(+:cnt_new)
			for(int i = cur_size; i < left_size; ++i) {
				int v = left_nodes[i];
				int d = dp_eta_deg(removed, pos_thread, now_thread, pre_thread, eta_pro[v], v, k);
				eta_Deg[v] = d;
				if (d < k) {
					// #pragma omp atomic capture
					// x = isdel_size++;
					// is_Del[x] = v;
					cnt_new ++;
					Buf[buf_size++] = v;
					if (buf_size == buf_cap){
						buf_size = 0;
						#pragma omp atomic capture
						x = isdel_size += buf_cap;
						x -= buf_cap;
						for (int h = 0; h < buf_cap; ++h)
							is_Del[x++] = Buf[h];
					}
				}
			}
		 	Double *pro_update_th = pro_update[pid];
			#pragma omp for schedule(dynamic) reduction(+:cnt_new)
			for (int i = 0; i < cur_size; ++i) {
				int v = left_nodes[i];
				//if (eta_Deg[v] < k) { // optimization of computing eat-degrees
					int update_cnt = 0, d = deg[v];
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						if (UB[u] == k) 
							pro_update_th[update_cnt++] = adj[v][j].p;
					}
					if (update_cnt) {
						memcpy(pre_thread, eta_pro[v], sizeof(Double) * k);
						d = apped_eta_deg(pro_update_th, update_cnt, now_thread, pre_thread, eta_pro[v], k);
						//d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, k);
						eta_Deg[v] = d;
						if (d >= k) continue;
					}
					
					// #pragma omp atomic capture
					// 	x = isdel_size++;
					// is_Del[x] = v;	
					cnt_new ++;
					Buf[buf_size++] = v;
					if (buf_size == buf_cap){
						buf_size = 0;
						#pragma omp atomic capture
						x = isdel_size += buf_cap;
						x -= buf_cap;
						for (int h = 0; h < buf_cap; ++h)
							is_Del[x++] = Buf[h];
					}
				//}
			}
			#pragma omp atomic capture
			x = isdel_size += buf_size;
			x -= buf_size;
			for (int h = 0; h < buf_size; ++h)
				is_Del[x++] = Buf[h];
			buf_size = 0;
		}
	
		while(isdel_size > 0) {
			remove_size = 0;
			#pragma omp parallel if (isdel_size > 3)
			{
				int pid = omp_get_thread_num();
				Double *pos_thread = pos_threads[pid];
				Double *pre_thread = pre_threads[pid];
				Double *now_thread = now_threads[pid];
				int Buf[128];
				int buf_cap = 128, buf_size = 0; 
				int x;
				#pragma omp for schedule(dynamic)
				for (int i = 0; i < isdel_size; ++i) {
					int v = is_Del[i];
					int d = deg[v];
					removed[v] = 1;
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						if (!removed[u] && coreness[u] < k && eta_Deg[u] >= k)
						if (!visited[u]){
							#pragma omp atomic capture
							x = visited[u]++;
							if (x == 0) {
								// #pragma omp atomic capture
								// x = remove_size++;
								// Removend[x] = u;
								Buf[buf_size++] = u;
								if (buf_size == buf_cap){
									buf_size = 0;
									#pragma omp atomic capture
									x = remove_size += buf_cap;
									x -= buf_cap;
									for (int h = 0; h < buf_cap; ++h)
										Removend[x++] = Buf[h];
								}
							}
						}
					}
				}
				#pragma omp atomic capture
				x = remove_size += buf_size;
				x -= buf_size;
				for (int h = 0; h < buf_size; ++h)
					Removend[x++] = Buf[h];
				buf_size = 0;
				isdel_size = 0;
				#pragma omp barrier
				#pragma omp for schedule(dynamic) reduction(+:cnt_new)
				for (int i = 0; i < remove_size; ++i) {
					int v = Removend[i];
					visited[v] = 0;
					int d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, k);
					if (d < k) {
						removed[v] = 1;
						// #pragma omp atomic capture
						// x = isdel_size++;
						// is_Del[x] = v;
						cnt_new ++;
						Buf[buf_size++] = v;
						if (buf_size == buf_cap){
							buf_size = 0;
							#pragma omp atomic capture
							x = isdel_size += buf_cap;
							x -= buf_cap;
							for (int h = 0; h < buf_cap; ++h)
								is_Del[x++] = Buf[h];
						}
					}
				}
				#pragma omp atomic capture
				x = isdel_size += buf_size;
				x -= buf_size;
				for (int h = 0; h < buf_size; ++h)
					is_Del[x++] = Buf[h];
				buf_size = 0;
			}
		}

		left_size_new = 0;
		for (int i = 0; i < left_size; ++i) {
			int v = left_nodes[i];
			if (!removed[v]) 
				coreness[v] = k;
			else 
				left_nodes[left_size_new++] = v;
			removed[v] = 0;
		}
		cnt_cores +=  left_size - left_size_new;
		left_size = left_size_new;
	}
	printf("Test iterator time: %f s\n", omp_get_wtime() - pt);
	printf("Uncertain core decomposition time: %f s\n", omp_get_wtime() - start_tm);
	printf("Node numbers: %d\n", (int)vertices[0].size() + overall_size);

	for (int i = 0; i < threads; ++i) {
		delete[] pos_threads[i];
		delete[] pre_threads[i];
		delete[] now_threads[i];
		delete[] pro_update[i];
	}
	delete[] pos_threads;
	delete[] pre_threads;
	delete[] now_threads;
	delete[] pro_update;
	delete[] probability;
	delete[] eta_pro;
}

//parallel binary/isometrics search, and update eta-degrees of unremoved nodes by examining whether a neighbor is removed in the current iteration.
void Uncertain_Core::Optimal_topdown_core_decomposition_parallel(vec_i &coreness, Double eta)
{
	double pt, start_tm = omp_get_wtime();
	this->eta = eta;
	coreness.clear(); 
	coreness.resize(n, 0);
	vec_i UB, UB_new(n, 0);
	int max_ub = 0, max_lb = 0, max_core = 0;

	max_ub = upper_bound_parallel_new(UB);
	printf("max_lb = %d, max_ub = %d \n", max_lb, max_ub);
	
	pt = omp_get_wtime();
	Double **pos_threads, **pre_threads, **now_threads;
	Double *pos, *pro_new, *pro_old;
	Double **pro_update = new Double*[threads];
	pos_threads = new Double*[threads];
	pre_threads = new Double*[threads];
	now_threads = new Double*[threads];
	for (int i = 0; i < threads; ++i) {
		pos_threads[i] = new Double[max_degree+1];
		pre_threads[i] = new Double[max_degree+1];
		now_threads[i] = new Double[max_degree+1];
		pro_update[i] = new Double[max_degree+1];
	}
	pos = pos_threads[0];
	pro_new = now_threads[0];
	pro_old = pre_threads[0];

	Double *probability, **eta_pro;
	probability = new Double[m];
	eta_pro = new Double*[n];
	eta_pro[0] = probability;
	for (int i = 0; i < n - 1; ++i) 
		eta_pro[i+1] = eta_pro[i] + deg[i];
	
	vec_i removed(n, 1), eta_Deg(n, 0);
	vec_i left_nodes(n), cur_deg(n, 0), cur_deg_new(n, 0);
	vector < vec_i > vertices(max_ub+1);
	for (int i = 0; i < n; ++i) {
		int ub = UB[i];
		vertices[ub].push_back(i);
		UB_new[i] = ub;
	}
	
	int del_size = 0, left_size = 0;
	int mid_core = 0, cur_core = max_ub+1;
	int cnt = 0;
	int init_minc = max_lb, init_maxc = max_ub;
	
	//Detecting (maxlb, eta)-core iteratively
	max_lb = max_ub;
	while (true) {
		max_lb /= 2;
		int cur_del_size = del_size;
		int index_level = 2;
		cnt = 0;
		#pragma omp parallel
		{
			int x, pid = omp_get_thread_num();
			Double *pos_thread = pos_threads[pid];
			Double *pre_thread = pre_threads[pid];
			Double *now_thread = now_threads[pid];
			#pragma omp for schedule(dynamic)
			for (int k = max_lb; k <= max_ub; ++k) {
				size_t size = vertices[k].size();
				#pragma omp atomic capture
				x = del_size += size;
				x -= size;
				for (int i = 0; i < size; ++i) {
					int v = vertices[k][i];
					removed[v] = 0;
					Del[x] = v;
					left_nodes[x++] = v;
				}
			}
			left_size = del_size;

			#pragma omp for schedule(dynamic, 4)
			for (int i = cur_del_size; i < del_size; ++i) {
				int v = Del[i];
				int d = dp_eta_deg(removed, pos_thread, now_thread, pre_thread, eta_pro[v], v, UB_new[v]);
				eta_Deg[v] = d;
				cur_deg[v] = d;
				if (d < max_lb) 
					UB_new[v] = max_lb - 1;
			}
			#pragma omp for reduction(+:cnt)
			for (int i = 0; i < del_size; ++i) {
				int v = Del[i];
				if (cur_deg[v] < max_lb) {
					removed[v] = index_level;
					++cnt;
				}
			}
		}
		cnt = del_size - cnt;

		//Removing nodes with eta-degree less then 'max_lb' iteratively
		bool flag = true;
		while (flag) {
			flag = false;
			#pragma omp parallel 
			{
				int pid = omp_get_thread_num();
				Double *pos_thread = pos_threads[pid];
				Double *pre_thread = pre_threads[pid];
				Double *now_thread = now_threads[pid];
				Double *update_thr = pro_update[pid];
				int x = 0;
				#pragma omp for schedule(dynamic, 4)
				for (int i = 0; i < del_size; ++i) {
					int v = Del[i];
					if (removed[v] || cnt < max_lb)  continue;
					int d = deg[v];
					bool flag_n = false;
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						if (removed[u] == index_level) {
							flag_n = true; break;
						}
					}
					if (flag_n) {
						d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, UB_new[v]);
						cur_deg[v] = d;
						if (d < max_lb) {
							flag = true;
							UB_new[v] = max_lb - 1;
							removed[v] = index_level + 1;
							if (++x == 4) {
								#pragma omp atomic
								cnt -= x;
								x = 0;
							}
						}
					}
				}
				#pragma omp atomic
					cnt -= x;
				if (cnt < max_lb) flag = false;
			}
			index_level++;
		}
	
		if (cnt >= max_lb) break;

		cnt = 0; max_ub = max_lb - 1; 
		int max_lb_new = max_lb / 2;
		#pragma omp parallel 
		{
			int pid = omp_get_thread_num();
			Double *pos_thread = pos_threads[pid];
			Double *pre_thread = pre_threads[pid];
			Double *now_thread = now_threads[pid];
			Double *update_thr = pro_update[pid];
			#pragma omp for schedule(dynamic, 2) reduction(+:cnt)
			for (int i = 0; i < del_size; ++i) {
				int v = Del[i];
				int d = deg[v];
				int update_size = 0;
				removed[v] = 0;
				for (int j = 0; j < d; ++j){
					int u = adj[v][j].u;
					int ubu = UB[u];
					if (ubu < max_lb && ubu >= max_lb_new)
						update_thr[update_size++] = adj[v][j].p;
				}
				if (update_size) {
					memcpy(pre_thread, eta_pro[v], sizeof(Double) * max_ub);
					d = apped_eta_deg(update_thr, update_size, now_thread, pre_thread, eta_pro[v], max_ub);
					eta_Deg[v] = d;
				}
				d = cur_deg[v] = eta_Deg[v];
				
				if (d < max_lb_new)
					UB_new[v] = max_lb_new - 1;
				else 
					UB_new[v] = max_ub;
			}
		}
	}
	printf("Time: %f s, lb_core=%d, cnt=%d\n", omp_get_wtime() - pt, max_lb, cnt);
	#pragma omp parallel for
	for (int i = 0; i < left_size; ++i) {
		int v = left_nodes[i];
		if (removed[v])
			removed[v] = 1;
	}

	vec_i deg_lb(n, 0);
	vector <vec_i>  sortednodes(max_ub+1);
	int del_size_new = 0;
	for (int i = 0; i < del_size; ++i) {
		int v = Del[i];
		if (removed[v]) continue;
		Del[del_size_new++] = v;
		coreness[v] = max_lb;
	}
	del_size = del_size_new;
	#pragma omp parallel
	{
		vec_i bin(max_ub+1);
		#pragma omp for schedule(dynamic, 4)
		for (int i = 0; i < del_size; ++i) {
			int v = Del[i];
			int d = reset_cur_deg(removed, cur_deg, v, max_lb, max_ub, bin);
			assert(d <= max_ub && d >= max_lb);
			cur_deg[v] = d;
			deg_lb[v] = d;
			cur_deg_new[v] = d;
		}
	}
	for (int i = 0; i < del_size; ++i) {
		int v = Del[i];
		sortednodes[cur_deg[v]].push_back(v);
	}

	//Detecting the maximum eta-core with the binary search 
	init_minc = max_lb+1; init_maxc = max_ub;
	while (init_minc <= init_maxc) {
		mid_core = (init_minc + init_maxc) / 2;
		int index_level = 2;
		int cnt_new = del_size;
		for (int i = 0; i < del_size; ++i) {
			int v  = Del[i];
			if (cur_deg_new[v] < mid_core) {
				removed[v] = index_level;
				--cnt_new;
			}
		}
		while (true) {
			bool flag = false;
			#pragma omp parallel 
			{
				int pid = omp_get_thread_num();
				Double *pos_thread = pos_threads[pid];
				Double *pre_thread = pre_threads[pid];
				Double *now_thread = now_threads[pid];
				Double *update_thr = pro_update[pid];
				int x = 0;
				bool flag_n = false;
				#pragma omp for schedule(dynamic, 4)
				for (int i = 0; i < del_size; ++i) {
					int v  = Del[i];
					int d = cur_deg_new[v];
					if (removed[v] || cnt_new < mid_core) continue;
				
					flag_n = false;
					d = deg[v];
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						if (removed[u] == index_level) {
							flag_n = true; break;
						}
					}
					if(flag_n) {
						d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, UB_new[v]);
						cur_deg_new[v] = d;
						if (d < mid_core){
							flag = true;
							UB_new[v] = mid_core - 1;
							removed[v] = index_level+1;
							#pragma omp atomic
							--cnt_new;
						}
					}
				}
			}
			index_level++;
			if (!flag || cnt_new < mid_core) break;
		}

		if (cnt_new < mid_core) {
			init_maxc = mid_core - 1;
			max_core = init_maxc;
			for (int i = 0; i < del_size; ++i) {
				int v  = Del[i];
				int d = cur_deg[v];
				cur_deg_new[v] = d;
				removed[v] = 0;
				UB_new[v] = min(UB_new[v], init_maxc);
			}
		}
		else {
			init_minc = mid_core+1;
			cnt = cnt_new;
			max_core = mid_core; del_size_new = 0;
			for (int i = 0; i < del_size; ++i) {
				int v  = Del[i];
				int d = cur_deg_new[v];
				cur_deg[v] = d;
				if (d < mid_core) continue;
				Del[del_size_new++] = v;
				coreness[v] = mid_core;
			}
			del_size = del_size_new;
		}
	}

	printf("Time: %f s, max_core=%d, cnt=%d\n", omp_get_wtime() - pt, max_core, cnt);
	for (int i = max_lb; i <= max_ub; ++i) {
		size_t size = sortednodes[i].size();
		for (int j = 0; j < size; ++j){
			int v = sortednodes[i][j];
			removed[v] = 2;
		}
	}
	
	// Computing eta-cores in the interval [maxlb, max_core-1]
	cur_core = max_core - 1;
	del_size = 0;
	int cnt_core = 0, cnt_cores= 0;
	int cnt_core_new = 0;
	int k = cur_core;
	while (k > max_lb) {
		if (k == cur_core) {
			if (cur_core == max_lb) break;
			cur_core -= 10;
			cur_core = max(cur_core, max_lb+1);
			del_size = 0; cnt_core = 0;
			int cnt_size = 0;
			for (int i = cur_core; i <= max_ub; ++i) {
				size_t size = sortednodes[i].size();
				size_t size_new = 0;
				for (int j = 0; j < size; ++j){
					int v = sortednodes[i][j];
					removed[v] = 0;
					if (coreness[v] > k) {
						cnt_core++;
						continue;
					}
					int d = deg_lb[v];
					cur_deg[v] = d;
					sortednodes[i][size_new++] = v;
					if (d < cur_core || UB_new[v] < cur_core) {
						removed[v] = 2;
						continue;
					}
					Del[del_size++] = v;
					cnt_size++;
				}
				sortednodes[i].resize(size_new);
			}
		
			int cnt_new = 0;
			int index_level = 2;
			while (true) {
				bool flag = false;
				#pragma omp parallel 
				{
					int pid = omp_get_thread_num();
					Double *pos_thread = pos_threads[pid];
					Double *pre_thread = pre_threads[pid];
					Double *now_thread = now_threads[pid];
					Double *update_thr = pro_update[pid];
					int x = 0;
					bool flag_n = false;
					#pragma omp for schedule(dynamic, 4) reduction(+:cnt_new)
					for (int i = 0; i < del_size; ++i) {
						int v  = Del[i];
						if (removed[v]) continue;
						flag_n = false;
						int d = deg[v];
						for (int j = 0; j < d; ++j) {
							int u = adj[v][j].u;
							if (removed[u] == index_level) {
								flag_n = true; break;
							}
						}
						if(flag_n) {
							d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, min(UB_new[v], k));
							cur_deg[v] = d;
							if (d < cur_core){
								flag = true;
								removed[v] = index_level+1;
								//UB_new[v] = cur_core - 1;
								cnt_new++;
							}
						}
					}
				}
				index_level++;
				if (!flag) break;
			}
			del_size_new = 0;
			for (int i = 0; i < del_size; ++i){
				int v = Del[i];
				if (!removed[v]) {
					if (coreness[v] < cur_core)
						coreness[v] = cur_core;
					Del[del_size_new++] = v;
				}
				removed[v] = 2;
			}
			del_size = del_size_new;
			cur_core--;
			cnt_core_new = 0;
		}
		else {
			if (k == max_lb) break;
			int index_level = 3;
			int cnt_new = 0;
			#pragma omp parallel for reduction(+:cnt_new)
			for (int i = 0; i < del_size; ++i) {
				int v = Del[i];
				removed[v] = 0;
				if (cur_deg[v] < k) {
					removed[v] = index_level;
					cnt_new++;
				}
			}
			bool flag = false;
			while (true) {
				flag = false;
				#pragma omp parallel 
				{
					int pid = omp_get_thread_num();
					Double *pos_thread = pos_threads[pid];
					Double *pre_thread = pre_threads[pid];
					Double *now_thread = now_threads[pid];
					#pragma omp for schedule(dynamic, 4) reduction(+: cnt_new)
					for (int i = 0; i < del_size; ++i) {
						int v = Del[i];
						if (removed[v] || coreness[v] >= k) continue;
						int d = deg[v];
						int flag_n = false;
						for (int j = 0; j < d; ++j) {
							int u = adj[v][j].u;
							if (removed[u] == index_level) {
								flag_n = true; break;
							}
						}
						if (flag_n) {
							d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, k);
							if (d < k) {
								removed[v] = index_level+1;
								UB_new[v] = k - 1;
								flag = true;
								++cnt_new;
							}
						}
					}
				}
				index_level++;
				if (!flag) break;
			}
			del_size_new = 0;
			for (int i = 0; i < del_size; ++i){
				int v = Del[i];
				if (!removed[v]) {
					coreness[v] = k;
					cnt_core_new++;
				}
				else {
					Del[del_size_new++] = v;
					removed[v] = 3;
				}
			}
			del_size = del_size_new;
			--k;
		}
	}
	printf("Time: %f s, maxlb=%d\n", omp_get_wtime() - pt, max_lb);

	//Computing eta-cores in the interval (0, maxlb - 1]
	int left_size_new = 0;
	int overall_size = left_size;
	cnt_cores = 0;
	cur_core = max_lb - 1;
	k = cur_core;
	while (k > 0){
		if (k == cur_core) {
			if (cur_core > 20) cur_core -= 10;
			else cur_core /= 2; 
			cur_core = max(cur_core, 1);
			
			for (int i = 0; i < left_size; ++i) {
				int v = left_nodes[i];
				removed[v] = 0;
				if (coreness[v] >= k)
					cnt_cores ++;
				else
					left_nodes[left_size_new++] = v;
			}
			left_size = left_size_new;
			int cur_size = left_size;
			for (int i = cur_core; i <= k; ++i){
				size_t size = vertices[i].size();
				int v; overall_size += size;
				for (int j = 0; j < size; ++j) {
					v = vertices[i][j];
					left_nodes[left_size++] = v;
					removed[v] = 0;
				}
			}
			int index_level = 2;
			#pragma omp parallel 
			{
				int pid = omp_get_thread_num();
				Double *pos_thread = pos_threads[pid];
				Double *pre_thread = pre_threads[pid];
				Double *now_thread = now_threads[pid];
				Double *update_thr = pro_update[pid];
				#pragma omp for schedule(dynamic, 8)
				for (int i = cur_size; i < left_size; ++i) {
					int v = left_nodes[i];
					eta_Deg[v] = dp_eta_deg(removed, pos_thread, now_thread, pre_thread, eta_pro[v], v, UB[v]);
				}
				#pragma omp for schedule(dynamic, 8)
				for (int i = 0; i < cur_size; ++i) {
					int v = left_nodes[i];
					int d = deg[v];
					int update_size = 0;
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						int ubu = UB[u];
						if (ubu <= k && ubu >= cur_core) {
							update_thr[update_size++] = adj[v][j].p;
						}
					}
					if (update_size) {
						memcpy(pre_thread, eta_pro[v], sizeof(Double) * k);
						d = apped_eta_deg(update_thr, update_size, now_thread, pre_thread, eta_pro[v], k);
						//d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, k);
						eta_Deg[v] = d;
					}
				}
				#pragma omp for
				for (int i = 0; i < left_size; ++i) {
					int v = left_nodes[i];
					int d = eta_Deg[v];
					cur_deg[v] = d;
					if (d < cur_core) {
						removed[v] = index_level;
						UB_new[v] = cur_core - 1;
					}
				}
			}
			bool flag = false;
			while (true) {
				flag = false;
				#pragma omp parallel 
				{
					int pid = omp_get_thread_num();
					Double *pos_thread = pos_threads[pid];
					Double *pre_thread = pre_threads[pid];
					Double *now_thread = now_threads[pid];
					Double *update_thr = pro_update[pid];
					#pragma omp for schedule(dynamic, 8)
					for (int i = 0; i < left_size; ++i) {
						int v = left_nodes[i];
						if (removed[v]) continue;
						int d = deg[v];
						int flag_n = false;
						for (int j = 0; j < d; ++j) {
							int u = adj[v][j].u;
							if (removed[u] == index_level) {
								flag_n = true; break;
							}
						}
						if (flag_n) {
							d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, UB_new[v]);
							cur_deg[v] = d;
							if (d < cur_core) {
								removed[v] = index_level+1;
								UB_new[v] = cur_core - 1;
								flag = true;
							}
						}
					}
				}
				index_level++;
				if (!flag) break;
			}
		
			left_size_new = 0; del_size = 0;
			for (int i = 0; i < left_size; ++i) {
				int v = left_nodes[i];
				if (!removed[v]) {
					coreness[v] = cur_core;
					Del[del_size++] = v; 
					//int d = reset_cur_deg(cur_deg, v, cur_core, k, bin);
					//cur_deg[v] = d;
				}
				removed[v] = 1;
			}
			--cur_core;
		}
		else {
			if (k == 1) break;
			int cnt_new = 0, index_level = 2;
			#pragma omp parallel for reduction(+:cnt_new)
			for (int i = 0; i < del_size; ++i) {
				int v = Del[i];
				if (cur_deg[v] < k) {
					removed[v] = index_level;
					cnt_new++;
				}
				else 
					removed[v] = 0;
			}
			
			bool flag = false;
			while (true) {
				flag = false;
				#pragma omp parallel 
				{
					int pid = omp_get_thread_num();
					Double *pos_thread = pos_threads[pid];
					Double *pre_thread = pre_threads[pid];
					Double *now_thread = now_threads[pid];
					#pragma omp for schedule(dynamic, 4) reduction(+: cnt_new)
					for (int i = 0; i < del_size; ++i) {
						int v = Del[i];
						if (removed[v] || coreness[v] >= k) continue;
						int d = deg[v];
						int flag_n = false;
						for (int j = 0; j < d; ++j) {
							int u = adj[v][j].u;
							if (removed[u] == index_level) {
								flag_n = true; break;
							}
						}
						if (flag_n) {
							d = recompute_eta_degree(removed, pos_thread, now_thread, pre_thread, v, k);
							if (d < k) {
								removed[v] = index_level+1;
								UB_new[v] = k - 1;
								flag = true;
								++cnt_new;
							}
						}
					}
				}
				index_level++;
				if (!flag) break;
			}
			#pragma omp parallel for
			for (int i = 0; i < del_size; ++i){
				int v = Del[i];
				if (!removed[v] && coreness[v] < k)
					coreness[v] = k;
			}
			//printf("cur_core=%d, core_size=%d\n", k, cnt_cores + del_size - cnt_new);
			--k;
		}
	}
	printf("Test iterator time: %f s\n", omp_get_wtime() - pt);
	printf("Uncertain core decomposition time: %f s\n", omp_get_wtime() - start_tm);
	printf("Node Numbers: %d\n", (int)vertices[0].size() + overall_size);

	for (int i = 0; i < threads; ++i) {
		delete[] pos_threads[i];
		delete[] pre_threads[i];
		delete[] now_threads[i];
		delete[] pro_update[i];
	}
	delete[] pos_threads;
	delete[] pre_threads;
	delete[] now_threads;
	delete[] pro_update;
	delete[] probability;
	delete[] eta_pro;
}
