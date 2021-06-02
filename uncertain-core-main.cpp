#include "uncertain-core.h"

void load_cores(const char *str, vec_i &cores)
{
	const char *txt = strstr(str, "txt");
	const char *bin = strstr(str, "bin");
	if (txt == NULL && bin == NULL) {
		printf("Usage: file format \"*.txt\" or \"*.bin\"\n");
		exit(1);
	}

	int n = 0;
	FILE *in = NULL;
	if (txt != NULL) in = fopen(str, "r");
	else in = fopen(str, "rb");

	if (in == NULL) {
		printf("No such file: %s\n", str);
		exit(1);
	}
	if (txt != NULL) {
		int x = fscanf(in, "%d", &n);
		printf("file=%s, n=%d\n", str, n);
		cores.resize(n);
		for (int i = 0; i < n; ++i)
			x = fscanf(in, "%d", &cores[i]);
	}
	else { 
		size_t x = fread(&n, sizeof(int), 1, in);
		printf("file=%s, n=%d\n", str, n);
		cores.resize(n);
		for (int i = 0; i < n; ++i)
			x = fread(&cores[i], sizeof(int), 1, in);
	}
	fclose(in);
}

void prinf_core(const char *str, vec_i &cores)
{
	const char *txt = strstr(str, "txt");
	const char *bin = strstr(str, "bin");
	if (txt == NULL && bin == NULL) {
		printf("Usage: file format \"*.txt\" or \"*.bin\"\n");
		exit(1);
	}
	int n = cores.size();
	FILE *in = NULL;
	if (bin != NULL) in = fopen(str, "wb");
	else in = fopen(str, "w");

	if (in == NULL) {
		printf("No such file: %s\n", str);
		exit(1);
	}
	if (bin != NULL) {
		fwrite(&n, sizeof(int), 1, in);
		for (int i = 0; i < n; ++i)
			fwrite(&cores[i], sizeof(int), 1, in);
	}
	else {
		//printf("n=%d\n", n);
		fprintf(in, "%d\n", n);
		for (int i = 0; i < n; ++i)
			fprintf(in, "%d\n", cores[i]);
	}
	fclose(in);
}

void compares(const vec_i &core_acc, const vec_i &core_inacc)
{
	int cnt = 0;
	int n = core_acc.size();
	long double sum_errs = 0, avg_err = 0;
	for (int i = 0; i < n; ++i){
		int a = abs(core_acc[i] - core_inacc[i]);
		if ( a != 0 && core_acc[i] != 0) {
			sum_errs += (double) a;
			++cnt;
		}
	}
	if (n != 0) avg_err = double (cnt) / double (n);

	printf("Average errors: %Lf, cnt errors: %d\n", avg_err, cnt);
}

int main(int argc, char *argv[])
{

	if (strcmp("create-bin", argv[1]) == 0){
		printf("create-bin\n");
		if (argc < 3) { printf("Error: please input the file name.\n"); return 0;}
		Graph g;
		g.creat_bin(argv[2]);
		return 0;
	}

	if (strcmp("scale", argv[1]) == 0){
		if (argc < 4) { printf("Error: please input the file name.\n"); return 0;}
		int voe = atoi(argv[2]);
		float scale = atof(argv[3]);
		printf("scalability: %.1f, voe=%d\n", scale, voe);
		Uncertain_Core uc;
		uc.read_bin(argv[4], voe, scale);
		
		int alg = argc > 5 ? atoi(argv[5]) : 6;
		Double eta =  argc > 6 ? atof(argv[6]) : 0.6;
		int threads = argc > 7 ? atoi(argv[7]) : 1;
		int prune = 0;
		vec_i coreness;
		uc.get_sorted_adj(alg);
		printf("eta=%Lf\n", eta);
		uc.set_eta(eta);
		uc.set_threads(threads);
		switch(alg) {
			case 4:
				printf("Alg: ImpBottomUp_core_decomposition_parallel\n");
				prune = argc > 8 ? atoi(argv[8]) : prune;
				if (prune == 0)
					printf("Prune: Topkcore\n");
				else if (prune == 1)
					printf("Prune: Betacore\n");
				else 
					printf("Prune: Hybrid\n");
				uc.ImpBottomUp_core_decomposition_parallel(coreness, eta, prune);
				break;
			case 5:
				printf("Alg: basic top-down search with lb\n");
				uc.Basic_topdown_core_decomposition_parallel(coreness, eta);
				break;
			case 6:
				printf("Alg: Optimal top-down with the partition search\n");
				uc.Optimal_topdown_core_decomposition_parallel(coreness, eta);
				break;
			default:
				break;
		}
		return 0;
	}
	
	Uncertain_Core uc;
	vec_i coreness, coreness1, LB, UB, LB1;
	uc.read_bin(argv[1]);
	int alg = argc > 2 ? atoi(argv[2]) : 6;
	Double eta =  argc > 3 ? atof(argv[3]) : 0.5;
	int threads = argc > 4 ? atoi(argv[4]) : 1;
	int print = argc > 5 ? atoi(argv[5]) : 0;
	int prune = 0;
	mp_bitcnt_t precision = 64;
	uc.get_sorted_adj(alg);
	printf("eta=%Lf\n", eta);
	double t = omp_get_wtime();
	uc.set_eta(eta);
	uc.set_threads(threads);
	switch(alg){
		case 1:
			printf("Alg: truncated_uncertain_core_decomposition\n");
			//precision = mpf_get_default_prec();
			precision = argc > 6 ? atoi(argv[6]) : precision;
			if (precision <= 64) {
				uc.truncated_uncertain_core_decomposition(coreness, eta);
				break;
			}
			else {
				mpf_set_default_prec(precision);
				printf("precision: %Zd\n", mpf_get_default_prec());
				uc.truncated_uncertain_core_decomposition_muti_precision(coreness, eta);
			}
			break;
		case 2:
			printf("Alg: Basic_precise_core_decomposition\n");
			uc.Basic_precise_core_decomposition(coreness, eta);
			break;
		case 3:
			printf("Alg: BottomUp_core_decomposition\n");
			prune = argc > 6 ? atoi(argv[6]) : prune;
			if (prune == 0)
				printf("Prune: Topkcore\n");
			else if (prune == 1)
				printf("Prune: Betacore\n");
			else 
				printf("Prune: Hybrid\n");
			uc.BottomUp_core_decomposition(coreness, eta, prune);
			break;
		case 4:
			printf("Alg: ImpBottomUp_core_decomposition_parallel\n");
			prune = argc > 6 ? atoi(argv[6]) : prune;
			if (prune == 0)
				printf("Prune: Topkcore\n");
			else if (prune == 1)
				printf("Prune: Betacore\n");
			else 
				printf("Prune: Hybrid\n");
			uc.ImpBottomUp_core_decomposition_parallel(coreness, eta, prune);
			break;
        case 5:
			printf("Alg: Basic top-down search with lb\n");
			uc.Basic_topdown_core_decomposition_parallel(coreness, eta);
			break;
		case 6:
			printf("Alg: Optimal top-down with the partition search\n");
			uc.Optimal_topdown_core_decomposition_parallel(coreness, eta);
			break;
		default:
			printf("Error: Please choose an algorithm in [1,6]\n");
			break;
	}
	printf("All time: %f s\n", omp_get_wtime() - t);
	if (print > 0) {
		char buf[128], outfile[128] = "";
		char *graph_file = argv[1];
		char *p = strrchr(graph_file, '/');
		if (p == NULL) p = graph_file;
		else p += 1;
		const char *filepath = "res-core/";
		if (access(filepath, 0) == -1) {
			mkdir(filepath, 0777);
		}
		strcat(outfile, filepath);
		strncat(outfile, p, strlen(p)-4);
		if (alg == 2) {
			sprintf(buf,"-acc-%d", (int) precision);
			strcat(outfile, buf);
		}
		if (print == 1)
			sprintf(buf,"-alg-%d-eta-%.3Lf-core.bin",alg,eta);
		else
			sprintf(buf,"-alg-%d-eta-%.3Lf-core.txt",alg,eta);
		strcat(outfile, buf);
		printf("outfile=%s\n",outfile);
		prinf_core(outfile, coreness);
	}

	return 0;
}
