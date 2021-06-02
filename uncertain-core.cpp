#include "uncertain-core.h"

Uncertain_Core::Uncertain_Core()
{
	eta = 1.0;
	threads = 1;
	buf_th_size = 128;

	adj_nbrs = NULL;
	Adj = NULL;
	leftc = NULL; leftn = NULL;
}

Uncertain_Core::~Uncertain_Core()
{
	if (adj_nbrs != NULL) delete[] adj_nbrs;
	if (Adj != NULL) delete[] Adj;
	if (leftc != NULL) delete[] leftc;
	if (leftn != NULL) delete[] leftn;
}

void Uncertain_Core::set_threads(int th)
{
	threads = th > 0 ? th : 1;
	omp_set_num_threads(threads);
	printf("Thread nums: %d\n", threads);
}

void Uncertain_Core::set_eta(Double eta)
{
	this->eta = eta;
	//printf("eta = %Lf\n", eta);
}

void Uncertain_Core::get_sorted_adj(int alg)
{
	adj_nbrs = new pair<Double, int>[m];
	Adj = new pair<Double, int>*[n];
	// for (Long i = 0; i < m; ++i)
	// {
	// 	adj_nbrs[i].first = data[i].p;
	// 	adj_nbrs[i].second = data[i].u;
	// }
	Adj[0] = adj_nbrs;
	for (int i = 0; i < n - 1; ++i)
		Adj[i+1] = Adj[i] + deg[i];
	for (int i = 0; i < n; ++ i) {
		for (int j = 0; j < deg[i]; ++j) {
			Adj[i][j].first = adj[i][j].p;
			Adj[i][j].second = adj[i][j].u;
		}
	}
	//#pragma omp parallel for schedule(dynamic, 16)
	for (int i = 0; i < n; ++i)	
		sort(Adj[i], Adj[i] + deg[i], greater< pair<Double, int> >());

	Del.resize(n); is_Del.resize(n);
	leftc = new int[n];
	leftn = new int[n];
}

int Uncertain_Core::eta_degree(Double *pro_new, Double *pro_old, int v)
{
	int d = deg[v];
	if (d <= 0) return 0;
	pro_new[0] = pro_old[0] = 1.0;
	for (int i = 1; i <= d; ++i)
		pro_new[i] = pro_old[i] = pro_old[i - 1] * (1 - adj[v][i - 1].p);
	pro_new[0] = pro_old[0] = pro_old[d];
	Double *pro_temp, pre_pro = 1.0, pro = 1.0;
	pro -= pro_new[0];
	if (pro < eta) return 0;

	for (int i = 1; i <= d; ++i)
	{
		for (int j = i; j <= d; ++j)
		{
			Double p = adj[v][j - 1].p;
			if (i == j)
				pro_new[j] = p * pre_pro;
			else
				pro_new[j] = p * pro_old[j - 1] + (1 - p) * pro_new[j - 1];
		}
		pre_pro = pro_new[i];
		pro_new[i] = pro_new[d];
		pro_old[i] = pro_new[i];
		pro -= pro_new[i];

		if (pro + 1e-16 < eta) return i;
		pro_temp = pro_old;
		pro_old = pro_new;
		pro_new = pro_temp;
	}

	pro_temp = pro_old;
	pro_old = pro_new;
	pro_new = pro_temp;
	return d;
}

int Uncertain_Core::eta_degree(Double *pro_new, Double *pro_old, int v, int ub)
{
	int cnt = 0, d = deg[v];
	if (d <= 0) return 0;
	Double p, pro = 1.0;
	Double *pro_temp, pre_pro = 1.0;
	pro_new[0] = pro_old[0] = 1.0;
	for (int i = 1; i <= d; ++i)
	{
		p = adj[v][i - 1].p;
		if (p + 1e-13 > 1.0) ++cnt;
		pro_new[i] = pro_old[i] = pro_old[i - 1] * (1 - p);
	}
	pro_new[0] = pro_old[0] = pro_old[d];
	pro -= pro_new[0];
	if (pro < eta) return 0;
	ub += cnt; ub = min(ub, d);

	for (int i = 1; i <= ub; ++i)
	{
		pro_new[i] = adj[v][i - 1].p * pre_pro;
		for (int j = i + 1; j <= d; ++j)
		{
			p = adj[v][j - 1].p;
			pro_new[j] = p * pro_old[j - 1] + (1 - p) * pro_new[j - 1];
		}
		pre_pro = pro_new[i];
		pro_new[i] = pro_new[d];
		pro_old[i] = pro_new[i];
		pro -= pro_new[i];
		if (pro + 1e-16 < eta)
			return i;
		pro_temp = pro_old;
		pro_old = pro_new;
		pro_new = pro_temp;
	}

	pro_temp = pro_old;
	pro_old = pro_new;
	pro_new = pro_temp;
	return ub;
}

int Uncertain_Core::eta_degree_mpf(mpf_t *pro_new, mpf_t *pro_old, int v, int ub)
{
	int cnt = 0, d = deg[v];
	if (d <= 0) return 0;
	mpf_t p, rp, pro, pre_pro, pro_tem, *pro_temp;
	mpf_inits(p, rp, pro, pre_pro, pro_tem, (mpf_ptr)0);
	mpf_set_ui(pro, 1);
	mpf_set_ui(pre_pro, 1);
	mpf_set_ui(pro_new[0], 1);
	mpf_set_ui(pro_old[0], 1);

	for (int i = 1; i <= d; ++i)
	{
		if (adj[v][i - 1].p + 1e-13 > 1.0) ++cnt;
		mpf_set_d(p, adj[v][i - 1].p);
		mpf_ui_sub(rp, 1, p);
		mpf_mul(pro_old[i], pro_old[i-1], rp);
		mpf_set(pro_new[i],pro_old[i]);
	}
	mpf_set(pro_new[0],pro_old[d]);
	mpf_set(pro_old[0],pro_new[0]);
	mpf_sub(pro, pro, pro_new[0]);
	if ( mpf_cmp_d(pro, eta) < 0) return 0;
	ub += cnt; ub = min(ub, d);

	for (int i = 1; i <= ub; ++i)
	{
		mpf_set_d(p, adj[v][i - 1].p);
		mpf_mul(pro_new[i], p, pre_pro);
		for (int j = i + 1; j <= d; ++j)
		{
			mpf_set_d(p, adj[v][j - 1].p);
			mpf_ui_sub(rp, 1, p);
			mpf_mul(pro_tem, pro_old[j - 1], p);
			mpf_mul(pro_new[j], pro_new[j - 1], rp);
			mpf_add(pro_new[j], pro_tem, pro_new[j]);
		}
		mpf_set(pre_pro, pro_new[i]);
		mpf_set(pro_new[i], pro_new[d]);
		mpf_set(pro_old[i], pro_new[i]);
		mpf_sub(pro, pro, pro_new[i]);
		if ( mpf_cmp_d(pro, eta) < 0) return i;
		pro_temp = pro_old;
		pro_old = pro_new;
		pro_new = pro_temp;
	}
	mpf_clears(p, pro, pre_pro, pro_tem, (mpf_ptr)0);

	pro_temp = pro_old;
	pro_old = pro_new;
	pro_new = pro_temp;
	return ub;
}

void Uncertain_Core::truncated_uncertain_core_decomposition(vec_i &coreness, Double eta)
{
	this->eta = eta;
	coreness.clear();
	coreness.resize(n, 0);
	vec_i UB;
	Double **probability = new Double *[n];
	Double *pro_new, *pro_old;
	pro_new = new Double[max_degree + 1];
	pro_old = new Double[max_degree + 1];
	double tm = omp_get_wtime();
	upper_bound(UB);
	int max_c = 0, max_ub = 0;
	for (int i = 0; i < n; ++i)
	{
		probability[i] = NULL;
		int d = eta_degree(pro_new, pro_old, i, UB[i]);
		if (d <= 0) continue;

		coreness[i] = d;
		probability[i] = new Double[d + 1];
		for (int j = 0; j <= d; ++j)
			probability[i][j] = pro_new[j];
		max_c = max(max_c, d);
		max_ub = max(max_ub, UB[i]);
	}
	delete[] pro_new;
	delete[] pro_old;
	printf("Init time: %lf s\n", omp_get_wtime() - tm);
	printf("Max init core: %d, Max ub: %d\n", max_c, max_ub);
	max_c = 0;

	int max_init_core = 0, max_core = 0, min_core = n;
	vec_i D, index(n);
	vector<vec_i> vertices(n);

	for (int i = 0; i < n; ++i)
	{
		int core = coreness[i];
		index[i] = (int)vertices[core].size();
		vertices[core].push_back(i);
		max_init_core = max(max_init_core, core);
	}
	int cnt = 0;
	for (int i = 0; i <= max_degree; ++i)
	{
		size_t size = vertices[i].size();
		for (int j = 0; j < size; ++j)
		{
			int v = vertices[i][j];
			int d = deg[v];
			cnt++;
			for (int k = 0; k < d; ++k)
			{
				int u = adj[v][k].u;
				Double p = adj[v][k].p;
				int core_u = coreness[u];
				if (core_u > i)
				{
					Double q = 1.0, *pro = probability[u];
					int core_new = core_u;
					if (p + 1e-13 > 1.0) {
						for (int l = 0; l < core_new; ++l)
							pro[l] = pro[l+1];
						core_new--;
					}
					else {
						pro[0] /= (1 - p);
						q -= pro[0];
						if (q + 1e-16 < eta) core_new = 0;
						for (int l = 1; l <= core_new; ++l)
						{
							pro[l] = (pro[l] - p * pro[l - 1]) / (1 - p);
							q -= pro[l];
							// if (q + 1e-16 < eta) {
							// 	core_new = l - 1;
							// 	break;
							// }
						}
						if (q + 1e-16 < eta)
							core_new--;
					}
					
					core_new = max(i, core_new);
					coreness[u] = core_new;

					if (core_new != core_u)
					{
						int pos = index[u];
						int pos_end = (int)vertices[core_u].size() - 1;
						int w = vertices[core_u][pos_end];

						index[u] = (int)vertices[core_new].size();
						vertices[core_new].push_back(u);
						if (w != u)
						{
							vertices[core_u][pos] = w;
							index[w] = pos;
						}
						vertices[core_u].resize(pos_end);
						size = vertices[i].size();
					}
				}
			}
		}
		if (size > 0) max_core = i;
	}

	printf("Uncertain core decomposition time: %lf s\n", omp_get_wtime() - tm);
	printf("Max uncertain coreness: %d\n", max_core);
	//printf("Removal vertices: %d\n", cnt);

	for (int i = 0; i < n; ++i)
		if (probability[i] != NULL)
			delete[] probability[i];
	delete[] probability;
}

void Uncertain_Core::truncated_uncertain_core_decomposition_muti_precision(vec_i &coreness, Double eta)
{
	this->eta = eta;
	coreness.clear();
	coreness.resize(n, 0);
	vec_i UB;
	int alloc_size = max_degree+1;
	mpf_t mpf_pro_new[alloc_size], mpf_pro_old[alloc_size];
	mpf_t **mpf_probability = new mpf_t*[n];
	for (int i = 0; i <= max_degree; ++i){
		mpf_inits(mpf_pro_new[i], mpf_pro_old[i], (mpf_ptr)0);
	}
	double tm = omp_get_wtime();
	upper_bound(UB);
	int max_c = 0, max_ub = 0;
	for (int i = 0; i < n; ++i)
	{	
		mpf_probability[i] = NULL;

		int d = eta_degree_mpf(mpf_pro_new, mpf_pro_old, i, UB[i]);
		if (d <= 0) continue;

		coreness[i] = d;
		mpf_probability[i] = new mpf_t[d + 1];
		for (int j = 0; j <= d; ++j) {
			mpf_init_set(mpf_probability[i][j], mpf_pro_new[j]);
		}
		
		max_c = max(max_c, d);
		max_ub = max(max_ub, UB[i]);
	}
	printf("Init time: %lf s\n", omp_get_wtime() - tm);
	printf("Max init core: %d, Max ub: %d\n", max_c, max_ub);
	max_c = 0;

	int max_init_core = 0, max_core = 0, min_core = n;
	vec_i index(n);
	vector<vec_i> vertices(n);

	for (int i = 0; i < n; ++i)
	{
		int core = coreness[i];
		index[i] = (int)vertices[core].size();
		vertices[core].push_back(i);
		max_init_core = max(max_init_core, core);
	}

	mpf_t mpf_p_, mpf_rp, mpf_q_, mpf_r_; 
	mpf_inits(mpf_p_, mpf_rp, mpf_q_, mpf_r_, (mpf_ptr) 0);
	for (int i = 0; i <= max_init_core; ++i)
	{
		size_t size = vertices[i].size();
		for (int j = 0; j < size; ++j)
		{
			int v = vertices[i][j];
			int d = deg[v];	
			for (int k = 0; k < d; ++k)
			{
				int u = adj[v][k].u;
				Double p = adj[v][k].p;
				int core_u = coreness[u];
				if (core_u > i)
				{
					mpf_set_si(mpf_q_, 1);
					mpf_t *pro = mpf_probability[u];
					int core_new = core_u;
					if (p + 1e-13 > 1.0) {
						for (int l = 0; l < core_new; ++l)
						{
							mpf_set(pro[l], pro[l+1]);
						}
						core_new--;
					}
					else {
						mpf_set_d(mpf_p_, p);
						mpf_set_d(mpf_rp, 1 - p);
						mpf_div(pro[0], pro[0], mpf_rp);
						mpf_sub(mpf_q_, mpf_q_, pro[0]);
						if (mpf_cmp_d(mpf_q_, eta) < 0) core_new = 0;
			
						for (int l = 1; l < core_new; ++l)
						{
							mpf_mul(mpf_r_, mpf_p_, pro[l-1]);
							mpf_sub(pro[l], pro[l], mpf_r_);
							mpf_div(pro[l],pro[l], mpf_rp);
							mpf_sub(mpf_q_, mpf_q_, pro[l]);
							// if (mpf_cmp_d(mpf_q_ , eta) < 0) {
							// 	core_new = l-1;
							// 	break;
							// }
						}
					}
					if (mpf_cmp_d(mpf_q_ , eta) < 0)
						core_new--;
					
					core_new = max(i, core_new);
					coreness[u] = core_new;

					if (core_new != core_u)
					{
						int pos = index[u];
						int pos_end = (int)vertices[core_u].size() - 1;
						int w = vertices[core_u][pos_end];

						index[u] = (int)vertices[core_new].size();
						vertices[core_new].push_back(u);
						if (w != u)
						{
							vertices[core_u][pos] = w;
							index[w] = pos;
						}
						vertices[core_u].resize(pos_end);
						size = vertices[i].size();
					}
				}
			}
		}
		if (size > 0) max_core = i;
	}
	mpf_clears(mpf_p_, mpf_rp, mpf_q_, mpf_r_, (mpf_ptr) 0);
	for (int i = 0; i < n; ++i)
	{	
		if (!mpf_probability[i])
			delete[] mpf_probability[i];
	}
	delete[] mpf_probability;

	printf("Uncertain core decomposition time: %lf s\n", omp_get_wtime() - tm);
	printf("Max uncertain coreness: %d\n", max_core);
}

//lower bound + upper bound to improve efficiency
void Uncertain_Core::lower_bound(vec_i &LB)
{
	if (LB.empty()) LB.resize(n);
	if (adj_nbrs == NULL || Adj == NULL) return;
	double tm = omp_get_wtime();
	vec_i index_topk(n);
	vec_d prob(n);
	vec_b visited(n, 0);
	for (int i = 0; i < n; ++i)
	{
		int d = deg[i];
		Double p = 1.0, q = 1.0;
		for (int j = 0; j < d; ++j)
		{
			q = p * Adj[i][j].first;
			if (q + 1e-16 < eta){
				d = j; break;
			}
			p = q;
		}
		prob[i] = p;
		LB[i] = d;
		index_topk[i] = d - 1;
	}

	vector<vec_i> vertices(n);
	vec_i core_index(n);
	int max_init_lb = 0, max_lb = 0;

	for (int i = 0; i < n; ++i)
	{
		int core = LB[i];
		core_index[i] = (int)vertices[core].size();
		vertices[core].push_back(i);
		max_init_lb = max(max_init_lb, core);
	}
	printf("Maximum init number: %d\n", max_init_lb);
	int cnt = 0;
	for (int i = 0; i <= max_init_lb; ++i)
	{
		size_t size = vertices[i].size();
		for (int j = 0; j < size; ++j)
		{
			int v = vertices[i][j];
			int d = deg[v];
			cnt++;
			visited[v] = true;
			for (int k = 0; k < d; ++k)
			{
				int u = adj[v][k].u;
				Double p = adj[v][k].p;

				int core_u = LB[u];
				if (core_u > i)
				{
					Double q = 1.0, pro = 1;
					int du = deg[u];
					int core_new = 0;
					for (int l = 0; l < du; ++l){
						if (!visited[Adj[u][l].second]) {
							pro *= Adj[u][l].first;
							if (pro + 1e-16 < eta)
								break;
							++core_new;
						}
					}
					core_new = max(i, core_new);
					assert(core_new <= core_u);
					LB[u] = core_new;

					if (core_new != core_u)
					{
						int pos = core_index[u];
						int pos_end = (int)vertices[core_u].size() - 1;
						int w = vertices[core_u][pos_end];

						core_index[u] = (int)vertices[core_new].size();
						vertices[core_new].push_back(u);
						if (w != u)
						{
							vertices[core_u][pos] = w;
							core_index[w] = pos;
						}
						vertices[core_u].resize(pos_end);
						size = vertices[i].size();
					}
				}
			}
		}
		if (size > 0)
		{
			max_lb = i;
			//printf("Iterator %d, letf_vert = %d\n", i, n - cnt);
		}
	}

	printf("LB decomposition time: %lf s\nMaximum lower bound: %d\n", omp_get_wtime() - tm, max_lb);
	//printf("Removal vertices: %d\n", cnt);
}

int Uncertain_Core::lower_bound_parallel(vec_i &LB, vec_i &UB)
{
	if (UB.empty()) {
		printf("Function:lower_bound_parallel; UB is empty\n");
		exit(1);
	}
	LB.clear();
	LB.resize(n, 0);

	if (adj_nbrs == NULL || Adj == NULL)
		return 0 ;

	vec_i index_topk(n), visited(n, 0);
	vec_d prob(n);
	
	Double **adj_removed = new Double*[n];
	Double *removed = new Double[m];
	adj_removed[0] = removed;
	for (int i = 0; i < n - 1; ++i)
		adj_removed[i+1] = adj_removed[i] + deg[i];

	double tm = omp_get_wtime();
	int max_init_lb = 0, max_lb = 0;
	int *temp, left_size = n, left_sizen = 0;
	//bin_sort(Del);
	#pragma omp parallel
	{
		int v, d, j;
		Double p, q;
		pair<Double, int> *adj_v;
		#pragma omp for schedule(dynamic, 16) reduction(max : max_init_lb)
		for (int i = 0; i < n; ++i)
		{
			v = i;
			p = 1.0; q = 1.0;
			d = UB[v];
			adj_v = Adj[v];
			for (j = 0; j < d; ++j)
			{
				q = p * adj_v[j].first;
				if (q + 1e-16 < eta) break;
				p = q;
			}
			d = j;
			prob[v] = p;
			LB[v] = d;
			max_init_lb = max(max_init_lb, d);
		}
	}
	#pragma omp parallel for 
	for (int i = 0; i < n; ++i)
	{
		leftc[i] = i;
		index_topk[i] = LB[i] - 1;
	}

	printf("Init LB time: %lf s\nMaximum init number: %d\n", omp_get_wtime() - tm, max_init_lb);
	//tm = omp_get_wtime();

	int cnt = 0, k = 0, half_th_size = buf_th_size / 2;
	while (left_size > 0)
	{
		int del_size = 0, del_new_size = 0;
		left_sizen = 0;
		//#pragma omp parallel for
		for (int i = 0; i < left_size; ++i)
		{
			int v = leftc[i];
			int lb = LB[v];
			int x;
			if (lb == k) {
				//#pragma omp atomic capture
				x = del_size++;
				Del[x] = v;
			}
			else if (lb > k) {
				//#pragma omp atomic capture
				x = left_sizen++;
				leftn[x] = v;
			}
		}
		left_size = left_sizen;
		temp = leftc;
		leftc = leftn;
		leftn = temp;
		if (left_size <= k)
		{
			left_size + del_size > 0 ? k : --k;
			for (int i = 0; i < left_size; ++i)
				LB[leftc[i]] = k;
			cnt += left_size + del_size;
			break;
		}

		#pragma omp parallel if (del_size > 10)
		{
			int buf_thread[128];
			int x, le_size = 0;
			while (true){
				del_new_size = 0;
				#pragma omp barrier
				if (del_size <= 0) break;
				#pragma omp for schedule(dynamic, 4) reduction(+:cnt)
				for (int i = 0; i < del_size; ++i){
					int v = Del[i];
					int d = deg[v];
					cnt++;
					visited[v] = 1;
					for (int j = 0; j < d; ++j) {
						int u = adj[v][j].u;
						int core_u = LB[u];
						if (core_u > k){
							int pos_index = index_topk[u];
							Double p = adj[v][j].p;
							if (pos_index < deg[u])
								if ((Adj[u][pos_index].first - p > 1e-16) ||
									(Adj[u][pos_index].first - p > -1e-16 && Adj[u][pos_index].second > v))
									continue;
							#pragma omp atomic capture
								x = visited[u]--;
							x = 0 - x;
							assert(x >= 0);
							adj_removed[u][x] = p;
							if (x == 0){ 
								buf_thread[le_size++] = u;
								if (le_size == buf_th_size){
									le_size = 0;
									#pragma omp atomic capture
									x = del_new_size += buf_th_size;
									x -= buf_th_size;
									for (int j = 0; j < buf_th_size; ++j)
										is_Del[x++] = buf_thread[j];
								}
								//#pragma omp atomic capture
								//x = del_new_size++;
								//is_Del[x] = u;
							}
						}
					}
				}
				#pragma omp atomic capture
				x = del_new_size += le_size;
				x -= le_size;
				for (int j = 0; j < le_size; ++j)
					is_Del[x++] = buf_thread[j];
				del_size = 0;
				le_size = 0;
				#pragma omp barrier
				#pragma omp for schedule(dynamic, 4)
				for (int i = 0; i < del_new_size; ++i){
					int u = is_Del[i];
					int len = 0 - visited[u];
					visited[u] = 0;
					for (int j = 0; j < len; ++j){
						int core_u = LB[u];
						int pos_index = index_topk[u];
						Double p = adj_removed[u][j];
						Double q = 1.0, pro = prob[u];
						pro /= p;
						if (pos_index < deg[u]){
							int index_new = ++pos_index;
							for (; index_new < deg[u]; ++index_new){
								int w = Adj[u][index_new].second;
								if (visited[w] <= 0){
									q = pro * Adj[u][index_new].first;
									if (q + 1e-16 < eta){
										--index_new;
										--core_u;
									}
									else
										pro = q;
									break;
								}
							}
							if (index_new >= deg[u])
								--core_u;
							index_topk[u] = index_new;
							prob[u] = pro;
						}
						else
							--core_u;
						core_u = max(k, core_u);
						LB[u] = core_u;
						if (core_u <= k){
							buf_thread[le_size++] = u;
							if (le_size == buf_th_size){
								le_size = 0;
								#pragma omp atomic capture
								x = del_size += buf_th_size;
								x -= buf_th_size;
								for (int h = 0; h < buf_th_size; ++h)
									Del[x++] = buf_thread[h];
							}
							// #pragma omp atomic capture
							// 	len = del_size++;
							// Del[len] = u;
							break;
						}
					}
				}
				#pragma omp atomic capture
				x = del_size += le_size;
				x -= le_size;
				for (int h = 0; h < le_size; ++h)
					Del[x++] = buf_thread[h];
				le_size = 0;
			}
		}
		//printf("Iterator %d, left_vert = %d\n", k, n - cnt);
		++k;
	}
	max_lb = k;
	printf("LB decomposition time: %lf s\nMaximum lower bound: %d\n", omp_get_wtime() - tm, max_lb);
	delete[] adj_removed;
	delete[] removed;
	return max_lb;
}

void Uncertain_Core::lower_bound_of_beta_fun(vec_i &LB)
{
	LB.clear(); LB.resize(n, 0);

	double tm = omp_get_wtime();
	int max_lb = 0, max_v = 0;
	vec_i v_deg(n);
	for (int i = 0; i < n; ++i) 
	{
		int d = deg[i];
		Double p = 0, pmin = Adj[i][d-1].first;
		if (d <= 0 || pmin <= 1e-10) {
			LB[i] = 0;
			continue;
		}
		int Ex =  d * pmin;
		double q = 1.0;
		if (Ex <= 0) ++Ex;
		p = gsl_sf_beta_inc(Ex, d - Ex + 1, pmin);
		if (p - 1e-16 > eta) {
			while( Ex < d) {
				++ Ex;
				p = gsl_sf_beta_inc(Ex, d - Ex + 1, pmin);
				if ( p + 1e-16 < eta) {
					--Ex; break;
				}
			}
		}
		else if (p + 1e-16 < eta){
			-- Ex;
			for (;Ex > 0; --Ex) {
				p = gsl_sf_beta_inc(Ex, d - Ex + 1, pmin);
				if ( p - 1e-16 > eta) {
					break;
				}
			}
		}
		LB[i] = Ex;
		max_v = Ex > max_lb ? i : max_v;
		max_lb = max(Ex, max_lb);
		if (Ex < 0) {
			printf("errors\n");
		}
	}
	int d = deg[max_v];
	printf("Init time: %lf s\n", omp_get_wtime() - tm);
	printf("Max init lb: %d, max_v=%d, d=%d\n", max_lb, max_v, d);
	double pmin = pmin = Adj[max_v][d-1].first;
	int Ex = deg[max_v] * pmin;

	int left_size = n, left_sizen = 0;
	vec_i DE(n);
	size_t re_size = 0;
	for (int i = 0; i < n; ++i) {
		leftc[i] = i;
		v_deg[i] = deg[i];
	}
	int k = 0;
	while (left_size > 0) {
		re_size = 0;
		left_sizen = 0;
		for (int i = 0; i < left_size; ++i) {
			int v = leftc[i];
			if (LB[v] == k)
				DE[re_size++] = v;
			else if (LB[v] > k)
				leftn[left_sizen++] = v;
		}
		left_size = left_sizen;
		int * temp = leftc;
		leftc = leftn;
		leftn = temp;
		if (left_size <= k) {
			(left_size + re_size) > 0 ? k : k--;
			for (int i = 0; i < left_size; ++i) 
				LB[leftc[i]] = k;
			break;
		}

		for (int i = 0; i < re_size; ++i)
		{
			int u, v = DE[i];
			int du, cu, d = deg[v];
			LB[v] = k;
			for (int j = 0; j < d; ++j) {
				u = adj[v][j].u;
				cu = LB[u];
				if (cu > k) {
					double p = 0, pmin = Adj[u][deg[u]-1].first;
					du = v_deg[u]--;
					if (du == cu) {
						if (--LB[u] == k)
							DE[re_size++] = u;
						continue;
					}
					p = gsl_sf_beta_inc(cu, du - cu, pmin);
					if (p + 1e-16 < eta)  {
						-- cu;
						LB[u] = cu;
						if (cu == k) DE[re_size++] = u;
					}
				}
			}
		}
		++k;
	}
	printf("Beta_fun time: %lf s\n", omp_get_wtime() - tm);
	printf("Maximum beta lb: %d\n", k);
}

void Uncertain_Core::lower_bound_of_beta_fun_parallel(vec_i &LB)
{
	LB.clear(); LB.resize(n, 0);

	double tm = omp_get_wtime();
	int max_lb = 0, max_v = 0;
	vec_i v_deg(n);
	#pragma omp parallel for schedule(dynamic, 8) reduction(max:max_lb)
	for (int i = 0; i < n; ++i) 
	{
		int d = deg[i];
		Double p = 0, pmin = Adj[i][d-1].first;
		if (d <= 0 || pmin <= 1e-10) {
			LB[i] = 0;
			continue;
		}
		int Ex =  d * pmin;
		double q = 1.0;
		if (Ex <= 0) ++Ex;
		p = gsl_sf_beta_inc(Ex, d - Ex + 1, pmin);
		if (p - 1e-16 > eta) {
			while( Ex < d) {
				++ Ex;
				p = gsl_sf_beta_inc(Ex, d - Ex + 1, pmin);
				if ( p + 1e-16 < eta) {
					--Ex; break;
				}
			}
		}
		else if (p + 1e-16 < eta){
			-- Ex;
			for (;Ex > 0; --Ex) {
				p = gsl_sf_beta_inc(Ex, d - Ex + 1, pmin);
				if ( p - 1e-16 > eta) {
					break;
				}
			}
		}
		LB[i] = Ex;
		max_lb = max(Ex, max_lb);
		if (Ex < 0) {
			printf("errors\n");
		}
	}
	printf("Init time: %lf s\n", omp_get_wtime() - tm);
	printf("Max init lb: %d\n", max_lb);

	int left_size = n, left_sizen = 0;
	vec_i DE(n), DE_New(n), visited(n, 0);
	size_t re_size = 0, re_new_size = 0;
	#pragma omp parallel for
	for (int i = 0; i < n; ++i) {
		leftc[i] = i;
		v_deg[i] = deg[i];
	}
	int k = 0;
	while (left_size > 0) {
		re_size = 0;
		left_sizen = 0;
		for (int i = 0; i < left_size; ++i) {
			int v = leftc[i];
			if (LB[v] == k)
				DE[re_size++] = v;
			else if (LB[v] > k)
				leftn[left_sizen++] = v;
		}
		left_size = left_sizen;
		int * temp = leftc;
		leftc = leftn;
		leftn = temp;
		if (left_size <= k) {
			(left_size + re_size) > 0 ? k : k--;
			for (int i = 0; i < left_size; ++i) 
				LB[leftc[i]] = k;
			break;
		}
		while (re_size > 0) {
			re_new_size = 0;
			#pragma omp parallel
			{
				int buf[128];
				int x, buf_len = 0;
				#pragma omp for schedule(dynamic, 4)
				for (int i = 0; i < re_size; ++i)
				{
					int u, v = DE[i];
					int du, cu, d = deg[v];
					LB[v] = k;
					for (int j = 0; j < d; ++j) {
						u = adj[v][j].u;
						cu = LB[u];
						if (cu > k) {
							#pragma omp atomic capture
							x = visited[u]++;
							if (x == 0){
								buf[buf_len++] = u;
								if (buf_len == buf_th_size){
									buf_len = 0;
									#pragma omp atomic capture
									x = re_new_size += buf_th_size;
									x -= buf_th_size;
									for (int j = 0; j < buf_th_size; ++j)
										DE_New[x++] = buf[j];
								}
								// #pragma omp atomic capture
								// x = re_new_size++;
								// DE_New[x] = u;
							}
						}
					}
				}
				#pragma omp atomic capture
				x = re_new_size += buf_len;
				x -= buf_len;
				for (int j = 0; j < buf_len; ++j)
					DE_New[x++] = buf[j];
				re_size = 0; buf_len = 0;
				#pragma omp barrier
				#pragma omp for schedule(dynamic, 4)
				for (int i = 0; i < re_new_size; ++i){
					int u = DE_New[i];
					int cu = LB[u];
					int cnt = visited[u];
					int x, du = v_deg[u];
					visited[u] = 0;
					v_deg[u] -= cnt;
					double p = 0, pmin = Adj[u][deg[u]-1].first;
					for (int j = 0; j < cnt; ++j, --du) {
						p = 0;
						if (du <= cu) {
							--cu;
							if (--LB[u] == k) {
								buf[buf_len++] = u;
								if (buf_len == buf_th_size){
									buf_len = 0;
									#pragma omp atomic capture
									x = re_size += buf_th_size;
									x -= buf_th_size;
									for (int j = 0; j < buf_th_size; ++j)
										DE[x++] = buf[j];
								}

								//#pragma omp atomic capture
								//x = re_size++;
								//DE[x] = u;
								break;
							}
							continue;
						}
						assert(cu>0);
						assert(cu<du);
						p = gsl_sf_beta_inc(cu, du - cu, pmin);
						if (p + 1e-16 < eta)  {
							LB[u] = --cu;
							if (cu == k) {
								buf[buf_len++] = u;
								if (buf_len == buf_th_size){
									buf_len = 0;
									#pragma omp atomic capture
									x = re_size += buf_th_size;
									x -= buf_th_size;
									for (int j = 0; j < buf_th_size; ++j)
										DE[x++] = buf[j];
								}
								// #pragma omp atomic capture
								// x = re_size++;
								// DE[x] = u;
								break;
							}
						}
					}
				}
				#pragma omp atomic capture
				x = re_size += buf_len;
				x -= buf_len;
				for (int j = 0; j < buf_len; ++j)
					DE[x++] = buf[j];
				buf_len = 0;
			}
		}
		++k;
	}

	printf("Beta_fun time: %lf s\n", omp_get_wtime() - tm);
	printf("Maximum beta lb: %d\n", k);
}

int Uncertain_Core::upper_bound(vec_i &UB)
{
	UB.clear();UB.resize(n, 0);
	double tm = omp_get_wtime();
	vector<int> index(max_degree+1), pos(n);
	int max_core = 0;
	for (int d, i = 0; i < n; ++i)
	{
		d = deg[i];
		UB[i] = d;
		index[d]++;
	}
	for (int d, i = 0, cnt = 0; i <= max_degree; ++i)
	{
		d = index[i];
		index[i] = cnt;
		cnt += d;
	}
	for (int d, i = 0; i < n; ++i)
	{
		d = deg[i];
		int p = index[d]++;
		Del[p] = i;
		pos[i] = p;
	}
	for (int i = max_degree; i > 0; --i)
		index[i] = index[i-1];
	index[0] = 0;
	int lv = -1, lu = -1;
	for (int i = 0; i < n; ++i) {
		int v = Del[i];
		int c = UB[v];
		int d = deg[v];
		for (int j = 0; j < d; ++j){
			int u = adj[v][j].u;
			int cu = UB[u];
			if (cu > c) {
				UB[u]--;
				int posu = pos[u];
				int indexu = index[cu]++;
				if (posu > indexu) {
					int w = Del[indexu];
					Del[indexu] = u;
					Del[posu] = w;
					pos[w] = posu;
					pos[u] = indexu;
				}
				//lv = v; lu = u;
			}
		}
	}
	max_core = UB[Del[n-1]];

	printf("Maximum upper bound: %d\n", max_core);
	printf("Upper bound time: %lf s\n", omp_get_wtime() - tm );
	// for (int i = 0; i < n; ++i) {
	// 	int ct = 0;
	// 	int vb = UB[i];
	// 	for (int j = 0; j < deg[i]; ++j) {
	// 		int u = adj[i][j].u;
	// 		if (UB[u] >= vb)
	// 			++ct; 
	// 	}
	// 	if (ct < vb){
	// 		printf("errors: UB[%d]=%d, ct=%d\n", i, UB[i], ct);
	// 	}
	// }
	return max_core;
}

int Uncertain_Core::upper_bound_parallel_new(vec_i &UB)
{
	UB.clear();
	UB.resize(n);
	double tm = omp_get_wtime();
	int *now, *next, *temp;
	int d_size = 0, d_sizen = 0, left_size = n, left_sizen = 0;
	int half_th_size = buf_th_size / 2;

	int **UB_threads;
	vec_i UB_size(threads);
	UB_threads = new int*[threads];
	for (int i = 0; i < threads; ++i)
		UB_threads[i] = new int[n];

	int max_core = 0, k = 0;
	#pragma omp parallel for
	for (int i = 0; i < n; ++i){
		leftc[i] = i;
		UB[i] = deg[i];
	}

	while (left_size > 0) {
		d_size = 0;
		left_sizen = 0;
		#pragma omp parallel
		{
			int pid = omp_get_thread_num();
			int buf_thread[128];
			int *UB_thread = UB_threads[pid];
			int ub_size = 0;
			int x, now_size = 0, le_size = half_th_size;
			#pragma omp for //schedule(dynamic, 4)
			for (int i = 0; i < left_size; ++i){
				int v = leftc[i];
				int d = UB[v];
				if (d == k){
					UB_thread[ub_size++] = v;
				}
				else if (d > k){
					buf_thread[le_size++] = v;
					if (le_size == buf_th_size){
						le_size = half_th_size;
						#pragma omp atomic capture
						x = left_sizen += half_th_size;
						x -= half_th_size;
						for (int j = half_th_size; j < buf_th_size; ++j)
							leftn[x++] = buf_thread[j];
					}
				}
			}
			UB_size[pid] = ub_size;
			#pragma omp atomic capture
			x = left_sizen += le_size - half_th_size;
			x -= le_size - half_th_size;
			for (int j = half_th_size; j < le_size; ++j)
				leftn[x++] = buf_thread[j];	
		}
		left_size = left_sizen;
		temp = leftn;
		leftn = leftc;
		leftc = temp;
		if (left_size <= k){
			d_size = 0;
			for (int i = 0; i < threads; ++i)
				d_size += UB_size[i];
			left_size + d_size > 0 ? k : --k;
			for (int i = 0; i < left_size; ++i)
				UB[leftc[i]] = k;
			break; 
		}
		#pragma omp parallel 
		{
			int pid = omp_get_thread_num();
			int *UB_thread = UB_threads[pid];
			int ub_size = UB_size[pid];
			int ub_size_old = ub_size;
			int x;
			for (int i = 0; i < ub_size; ++i)
			{
				int u, v = UB_thread[i];
				int d = deg[v];
				int cu;
				UB[v] = k;
				for (int j = 0; j < d; ++j) {
					u = adj[v][j].u;
					cu = UB[u];
					if (cu > k) {
						#pragma omp atomic capture
						x = --UB[u];
						if (x == k)
							UB_thread[ub_size++] = u;
					}
				}
			}
			for (int i = ub_size_old; i < ub_size; ++i)
				UB[UB_thread[i]] = k;
		}
	
		++k;
	}
	max_core = k;
	for (int i = 0; i < threads; ++i)
		delete[] UB_threads[i];
	delete[] UB_threads;
	printf("Maximum upper bound: %d\n", max_core);
	printf("Upper bound time: %lf s\n", omp_get_wtime() - tm );
	return max_core;
}

int Uncertain_Core::recompute_eta_degree(vec_b &is_removed, vec_i &pos, Double *pro_new, Double *pro_old, int v, int ub)
{
	int u, pos_size = 0, d = deg[v];
	if (d <= 0) return 0;
	Double *pro_temp, p, pre_pro, pro = 1.0;
	pro_old[0] = pro_new[0] = 1.0;
	for (int i = 0; i < d; ++i)
	{
		u = adj[v][i].u;
		if (!is_removed[u])
		{
			pos[pos_size++] = i;
			p = adj[v][i].p;
			pro_new[pos_size] = pro_old[pos_size] = pro_old[pos_size - 1] * (1 - p);
		}
	}
	d = ub;
	pre_pro = pro_new[0];
	pro_new[0] = pro_old[0] = pro_old[pos_size];
	pro -= pro_new[0];
	if (pro + 1e-15 < eta) return 0;
	for (int i = 1; i < ub; ++i)
	{
		pro_new[i] = pre_pro * adj[v][pos[i - 1]].p;
		for (int j = i + 1; j <= pos_size; ++j)
		{
			p = adj[v][pos[j - 1]].p;
			pro_new[j] = pro_old[j - 1] * p + pro_new[j - 1] * (1 - p);
		}
		pre_pro = pro_new[i];
		pro -= pro_new[pos_size];
		if (pro + 1e-15 < eta)
		{
			d = i;
			break;
		}
		pro_temp = pro_new;
		pro_new = pro_old;
		pro_old = pro_temp;
	}
	pro_temp = pro_new;
	pro_new = pro_old;
	pro_old = pro_temp;
	return d;
}

int Uncertain_Core::recompute_eta_degree(vec_i &is_removed, Double *pos, Double *pro_new, Double *pro_old, int v, int ub)
{
	int pos_size = 0, d = deg[v];
	if (d <= 0) return 0;
	int u;
	Double *pro_temp, p, pre_pro, pro = 1;
	pro_old[0] = pro_new[0] = 1;
	for (int i = 0; i < d; ++i)
	{
		u = adj[v][i].u;
		if (is_removed[u] == 0)
		{
			p = adj[v][i].p;
			pos[pos_size++] = p;
			pro_new[pos_size] = pro_old[pos_size] = pro_old[pos_size - 1] * (1 - p);
		}
	}
	d = ub;
	pre_pro = pro_new[0];
	pro_new[0] = pro_old[0] = pro_old[pos_size];
	pro -= pro_new[0];
	if (pro + 1e-15 < eta) return 0;
	for (int i = 1; i < ub; ++i)
	{
		pro_new[i] = pre_pro * pos[i-1];
		for (int j = i + 1; j <= pos_size; ++j)
		{
			p = pos[j - 1];
			pro_new[j] = pro_old[j - 1] * p + pro_new[j - 1] * (1 - p);
		}
		pre_pro = pro_new[i];
		pro -= pro_new[pos_size];
		if (pro + 1e-15 < eta)
			return i;
		
		pro_temp = pro_new;
		pro_new = pro_old;
		pro_old = pro_temp;
	}
	return d;
}

int Uncertain_Core::recompute_eta_degree_mpf(vec_i &is_removed, int *pos, mpf_t *pro_new, mpf_t *pro_old, int v, int UB_v)
{
	int pos_size = 0, d = deg[v];
	if (d <= 0) return 0;
	mpf_t p, rp, pro, pre_pro, pro_tem, *pro_temp;
	mpf_inits(p, rp, pro, pre_pro, pro_tem, (mpf_ptr)0);
	mpf_set_ui(pro, 1);
	mpf_set_ui(pre_pro, 1);
	mpf_set_ui(pro_new[0], 1);
	mpf_set_ui(pro_old[0], 1);
	Double *pro_new_d, *pro_old_d, *pro_temp_d;
	Double pro_d = 1.0, pre_pro_d = 1.0, p_d;
	pro_new_d = new Double [d+1];
	pro_old_d = new Double [d+1];
	pro_new_d[0] = pro_old_d[0] = 1.0;

	for (int i = 0; i < d; ++i)
	{
		int u = adj[v][i].u;
		if (!is_removed[u])
		{
			pos[pos_size++] = i;
			mpf_set_d(p, adj[v][i].p);
			mpf_ui_sub(rp, 1, p);
			mpf_mul(pro_old[pos_size], pro_old[pos_size-1], rp);
			mpf_set(pro_new[pos_size], pro_old[pos_size]);
		}
	}
	mpf_set(pro_new[0],pro_old[pos_size]);
	mpf_set(pro_old[0],pro_new[0]);
	mpf_sub(pro, pro, pro_new[0]);
	if ( mpf_cmp_d(pro, eta) < 0) return 0;
	for (int i = 1; i < UB_v; ++i)
	{
		mpf_set_d(p, adj[v][pos[i - 1]].p);
		mpf_mul(pro_new[i], p, pre_pro);
		pro_new_d[i] = pre_pro_d * adj[v][pos[i - 1]].p;
		for (int j = i + 1; j <= pos_size; ++j)
		{
			mpf_set_d(p, adj[v][pos[j - 1]].p);
			mpf_ui_sub(rp, 1, p);
			mpf_mul(pro_tem, pro_old[j - 1], p);
			mpf_mul(pro_new[j], pro_new[j - 1], rp);
			mpf_add(pro_new[j], pro_tem, pro_new[j]);
		}
		mpf_set(pre_pro, pro_new[i]);
		mpf_set(pro_new[i], pro_new[pos_size]);
		mpf_set(pro_old[i], pro_new[i]);
		mpf_sub(pro, pro, pro_new[i]);
		if ( mpf_cmp_d(pro, eta) < 0) return i;

		pro_temp = pro_old;
		pro_old = pro_new;
		pro_new = pro_temp;
	}
	mpf_clears(p, pro, pre_pro, pro_tem, (mpf_ptr)0);
	delete[] pro_new_d;
	delete[] pro_old_d;
	pro_temp = pro_old;
	pro_old = pro_new;
	pro_new = pro_temp;
	return UB_v;
}

//The basic precise uncertain core decomposition algorithm
void Uncertain_Core::Basic_precise_core_decomposition(vec_i &coreness, Double eta)
{
	this->eta = eta;
	coreness.clear();
	coreness.resize(n, 0);
	vec_i LB, UB;
	int max_init_core = 0;
	Double *pro_now, *pro_pre;
	double tm = omp_get_wtime();
	pro_now = new Double[max_degree + 1]();
	pro_pre = new Double[max_degree + 1]();
	upper_bound(UB);
	for (int i = 0; i < n; ++i)
	{
		int d = eta_degree(pro_now, pro_pre, i, UB[i]);
		coreness[i] = d;
		max_init_core = max(max_init_core, d);
	}
	printf("Init time: %lf s\n", omp_get_wtime() - tm);

	vector< vec_i > vertices(max_init_core + 1);
	vec_b removed(n, false);
	vec_i index(n), dedeg(n), pos(max_degree);
	for (int i = 0; i < n; ++i)
	{
		int d = coreness[i];
		index[i] = (int)vertices[d].size();
		vertices[d].push_back(i);
	}
	int max_core = 0, cnt = 0;

	for (int i = 0; i <= max_init_core; ++i)
	{
		size_t size = vertices[i].size();
		for (int j = 0; j < size; ++j)
		{
			int d, v = vertices[i][j];
			++cnt;
			d = deg[v];
			removed[v] = true;
			for (int k = 0; k < d; ++k)
			{
				int u = adj[v][k].u;
				int core_u = coreness[u];
				if (core_u > i)
				{
					int core_new = core_u;
					core_new = recompute_eta_degree(removed, pos, pro_now, pro_pre, u, UB[u]);
					core_new = max(core_new, i);
					coreness[u] = core_new;

					if (core_new != core_u)
					{
						int pos = index[u];
						int pos_end = (int)vertices[core_u].size() - 1;
						int w = vertices[core_u][pos_end];

						index[u] = (int)vertices[core_new].size();
						vertices[core_new].push_back(u);
						if (w != u)
						{
							vertices[core_u][pos] = w;
							index[w] = pos;
						}
						vertices[core_u].resize(pos_end);
						size = vertices[i].size();
					}
				}
			}
		}
		if (size > 0)
		{
			max_core = i;
		}
	}
	delete[] pro_now;
	delete[] pro_pre;

	printf("Uncertain core dp time: %lf s\n", omp_get_wtime() - tm);
	printf("Max uncertain coreness: %d\n", max_core);
	//printf("Removal vertices: %d\n", cnt);
}

//The basic bottom-up uncertain core decomposition algorithm
void Uncertain_Core::BottomUp_core_decomposition(vec_i &coreness, Double eta, int prune)
{
	this->eta = eta;
	coreness.clear();
	coreness.resize(n, 0);
	vec_i LB, LB1, UB;
	int max_init_core = 0;
	Double *pro_now, *pro_pre;
	double tm = omp_get_wtime();
	pro_now = new Double[max_degree + 1]();
	pro_pre = new Double[max_degree + 1]();
	upper_bound_parallel_new(UB);
	if (prune == 0)
		lower_bound_parallel(LB, UB);
	else if (prune == 1)
		lower_bound_of_beta_fun_parallel(LB);
	else {
		//Hybrid
		lower_bound_parallel(LB, UB);
		lower_bound_of_beta_fun_parallel(LB1);
		for (int i = 0; i < n; ++i)
			LB[i] = max(LB[i], LB1[i]);
	}

	printf("LB+UB time: %lf s\n", omp_get_wtime() - tm);

	vector<vec_i> vertices(max_degree + 1);
	vec_b visited(n, false), removed(n, false);
	vec_i index(n), dedeg(n), pos(max_degree);

	for (int i = 0; i < n; ++i)
	{
		int d = LB[i];
		index[i] = (int)vertices[d].size();
		vertices[d].push_back(i);
	}
	int max_core = 0, cnt = 0;

	for (int i = 0; i <= max_degree; ++i)
	{
		size_t size = vertices[i].size();
		for (int j = 0; j < size; ++j)
		{
			int d, v = vertices[i][j];
			if (!visited[v])
			{
				d = recompute_eta_degree(removed, pos, pro_now, pro_pre, v, UB[v]);
				visited[v] = true;
				coreness[v] = max(d, i);
				if (d > i)
				{
					index[v] = (int)vertices[d].size();
					vertices[d].push_back(v);
					continue;
				}
			}
			int core = coreness[v];
			d = deg[v];
			++cnt;
			removed[v] = true;
			for (int k = 0; k < d; ++k)
			{
				int u = adj[v][k].u;
				int core_u = coreness[u];
				if (visited[u] && core_u > core)
				{
					int core_new = core_u;
					if (core_u + dedeg[u] >= deg[u])
						core_new = deg[u] - dedeg[u] - 1;

					core_new = recompute_eta_degree(removed, pos, pro_now, pro_pre, u, UB[u]);
					core_new = max(core_new, i);
					coreness[u] = core_new;

					if (core_new != core_u)
					{
						int pos = index[u];
						int pos_end = (int)vertices[core_u].size() - 1;
						int w = vertices[core_u][pos_end];

						index[u] = (int)vertices[core_new].size();
						vertices[core_new].push_back(u);
						if (w != u)
						{
							vertices[core_u][pos] = w;
							index[w] = pos;
						}
						vertices[core_u].resize(pos_end);

						size = vertices[i].size();
					}
				}
				--dedeg[u];
			}
		}
		if (size > 0)
		{
			max_core = i;
		}
	}
	delete[] pro_pre;
	delete[] pro_now;

	printf("Basic core decomposition time: %lf s\n", omp_get_wtime() - tm);
	printf("Max uncertain coreness: %d\n", max_core);
	//printf("Removal vertices: %d\n", cnt);
}

////The bottom-up uncertain core decomposition algorithm with lazy updating
void Uncertain_Core::ImpBottomUp_core_decomposition_parallel(vec_i &coreness, Double eta, int prune)
{
	double pt, start_tm = omp_get_wtime();
	this->eta = eta;
	coreness.clear();
	coreness.resize(n, -1);
	vec_i visited(n, 0), removed(n, 0), last_deg(n);
	Double **pos = new Double*[threads];
	int *temp, left_size = n, left_sizen = 0;

	Double **pre_thread, **now_thread;
	pre_thread = new Double *[threads];
	now_thread = new Double *[threads];
	for (int i = 0; i < threads; ++i)
	{
		pre_thread[i] = new Double[max_degree + 1];
		now_thread[i] = new Double[max_degree + 1];
		pos[i] = new Double [max_degree+1];
	}
	vec_i LB, LB1, UB;
	int max_init_core = 0, max_core = 0, cnt = 0;
	pt = omp_get_wtime();
	upper_bound_parallel_new(UB);
	if (prune == 0)
		lower_bound_parallel(LB, UB);
	else if (prune == 1)
		lower_bound_of_beta_fun_parallel(LB);
	else {
		//Hybrid
		lower_bound_parallel(LB, UB);
		lower_bound_of_beta_fun_parallel(LB1);
		for (int i = 0; i < n; ++i)
			LB[i] = max(LB[i], LB1[i]);
	}
	printf("Parallel LB+UB time: %lf s\n", omp_get_wtime() - pt);
	//return ;

	pt = omp_get_wtime();
	#pragma omp parallel for
	for (int i = 0; i < n; ++i){
		leftc[i] = i;
		last_deg[i] = deg[i];
	}
	int k = 0;
	while (left_size > 0)
	{
		int del_size = 0, is_del_size = 0;
		left_sizen = 0;
		//#pragma omp parallel for
		for (int i = 0; i < left_size; ++i)
		{
			int v = leftc[i];
			if (removed[v] == 0)
			{
				int lv = LB[v];
				int cv = coreness[v];
				int x = 0;
				if (lv == k || cv == k) {
					//#pragma omp atomic capture
					x = del_size++;
					Del[x] = v;
				}
				if (cv > k || cv == -1) {
					//#pragma omp atomic capture
					x = left_sizen++;
					leftn[x] = v;
				}
			}
		}
		left_size = left_sizen;
		temp = leftc;
		leftc = leftn;
		leftn = temp;
		if (left_size <= k)
		{
			//#pragma omp parallel for
			for (int i = 0; i < left_size; ++i)
				coreness[leftc[i]] = k;
			max_core = (left_size + del_size) > 0 ? k : k - 1;
			cnt += left_size + del_size;
			break;
		}

		#pragma omp parallel if (del_size > 5)
		{
			int pid = omp_get_thread_num();
			int buf_thread[128];
			Double *thr_pos = pos[pid];
			Double *thr_pro_new = pre_thread[pid];
			Double *thr_pro_old = now_thread[pid];
			int x, le_size = 0, buf_thread_size = 128;
			#pragma omp for schedule(dynamic, 4)
			for (int i = 0; i < del_size; ++i)
			{
				int v = Del[i];
				int d = coreness[v];
				if (d < 0)
				{
					d = recompute_eta_degree(removed, thr_pos, thr_pro_new, thr_pro_old, v, UB[v]);
					d = max(d, k);
					coreness[v] = d;
				}
				if (d <= k)
				{
					removed[v] = 1;
					#pragma omp atomic capture
						x = is_del_size++;
					is_Del[x] = v;
				}
			}

			while (true)
			{
				del_size = 0;
				#pragma omp barrier
				if (is_del_size <=0 ) break;
				#pragma omp for schedule(dynamic, 4) reduction(+: cnt)
				for (int i = 0; i < is_del_size; ++i)
				{
					int v = is_Del[i];
					int v_size = deg[v];
					for (int j = 0; j < v_size; ++j)
					{
						int u = adj[v][j].u;
						if (coreness[u] > k)
						{
							#pragma omp atomic capture
								x = visited[u]++;
							if (x == 0)
							{
								buf_thread[le_size++] = u;
								if (le_size == buf_thread_size){
									le_size = 0;
									#pragma omp atomic capture
									x = del_size += buf_thread_size;
									x -= buf_thread_size;
									for (int h = 0; h < buf_thread_size; ++h)
										Del[x++] = buf_thread[h];
								}
								//#pragma omp atomic capture
								//	x = del_size++;
								//Del[x] = u;
							}
						}
					}
					++cnt;
				}
				is_del_size = 0;
				#pragma omp atomic capture
				x = del_size += le_size;
				x -= le_size;
				for (int h = 0; h < le_size; ++h)
					Del[x++] = buf_thread[h];
				le_size = 0;
				#pragma omp barrier
				#pragma omp for schedule(dynamic, 4)
				for (int i = 0; i < del_size; ++i)
				{
					int v = Del[i];
					int d = recompute_eta_degree(removed, thr_pos, thr_pro_new, thr_pro_old, v, UB[v]);
					coreness[v] = max(d, k);
					visited[v] = 0;
					if (d <= k)
					{
						removed[v] = 1;
						// buf_thread[le_size++] = v;
						// if (le_size == buf_thread_size){
						// 	le_size = 0;
						// 	#pragma omp atomic capture
						// 	x = is_del_size += buf_thread_size;
						// 	x -= buf_thread_size;
						// 	for (int h = 0; h < buf_thread_size; ++h)
						// 		is_Del[x++] = buf_thread[h];
						// }
						#pragma omp atomic capture
							x = is_del_size++;
						is_Del[x] = v;
					}
				}
				// #pragma omp atomic capture
				// x = is_del_size += le_size;
				// x -= le_size;
				// for (int h = 0; h < le_size; ++h)
				// 	is_Del[x++] = buf_thread[h];
				// le_size = 0;
			}
		}
		//printf("Iterator %d, left-size = %d\n", k, n - cnt);
		++k;
	}
	for (int i = 0; i < threads; ++i)
	{
		delete[] pre_thread[i];
		delete[] now_thread[i];
		delete[] pos[i];
	}
	delete[] pre_thread;
	delete[] now_thread;
	delete[] pos;

	printf("Parallel iterator time: %lf s\n", omp_get_wtime() - pt);
	printf("Parallel core decomposition time: %lf s\n", omp_get_wtime() - start_tm);
	printf("Max uncertain coreness: %d\n", max_core);
}