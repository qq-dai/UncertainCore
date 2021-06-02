#pragma once
#ifndef GRAPH_H
#define GRAPH_H

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <omp.h>
#include <float.h>

#include <vector>
#include <string>
#include <algorithm>

using namespace std;

#define Int int
#define Long long long int
#define Double long double
#define Char char

struct edge
{
	Int u; double p;
	Int re; //The index of the reverse edge 
};
#define Pairs struct edge

class Graph
{
public:
	Int n, max_degree, *deg;
	Long m, *offs;
	Pairs *data, **adj;

	Graph();
	~Graph();

	void read_bin(const string &infile);
	void read_bin(const string &infile, int voe, float scale);
	void creat_bin(const string &infile);

	Int get_nm() { return n; }
	Long get_em() { return m; }
	Int *get_deg() { return deg; }
	Long *get_offs() { return offs; }
	Pairs **get_adj() { return adj; }
};

#endif // !GRAPH_H

