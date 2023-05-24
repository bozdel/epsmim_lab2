#include <stdio.h>

#include <emmintrin.h>
#include <immintrin.h>


/*
Ny--->
Nx
|
|
v

a b c d
e f g h

*/

void print_arr(double *arr, int Nx, int Ny) {
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			printf("%f ", arr[i * Ny + j]);
		}
		printf("\n");
	}
}

void print_mm(__m128d reg) {
	printf("%f %f\n", ((double*)&reg)[0], ((double*)&reg)[1]);
}

// #define _mm_loadu_pd _mm_load_pd

#define SIZE 1024

static inline void _process2(__m128d *phase_h_lwrl_p, __m128d *phase_h_mdll_p, double *phase_lwr, double *phase_mdl) {

	double res[10] = { 0 };



	// __m128d phase_h_lwrl = _mm_loadu_pd(phase_lwr);
	__m128d phase_h_lwrl = *phase_h_lwrl_p;
	__m128d phase_h_lwrr = _mm_loadu_pd(phase_lwr + 2);
	*phase_h_lwrl_p = phase_h_lwrr;
	// __m128d phase_h_mdll = _mm_loadu_pd(phase_mdl);
	__m128d phase_h_mdll = *phase_h_mdll_p;
	__m128d phase_h_mdlr = _mm_loadu_pd(phase_mdl + 2);
	*phase_h_mdll_p = phase_h_mdlr;

	// print_mm(phase_h_lwrl);
	// print_mm(phase_h_lwrr);
	// print_mm(phase_h_mdll);
	// print_mm(phase_h_mdlr);


	__m128d phase_sum_vl = _mm_add_pd(phase_h_lwrl, phase_h_mdll);
	__m128d phase_sum_vr = _mm_add_pd(phase_h_lwrr, phase_h_mdlr);

	// print_mm(phase_sum_vl);
	// print_mm(phase_sum_vr);

	__m128d phase_sum_hu = _mm_hadd_pd(phase_h_lwrl, phase_h_lwrr);
	__m128d phase_sum_hd = _mm_hadd_pd(phase_h_mdll, phase_h_mdlr);

	// print_mm(phase_sum_hu);
	// print_mm(phase_sum_hd);

	__m128d phase_lwr_mid = _mm_set_pd( ((double*)&phase_h_lwrr)[0], ((double*)&phase_h_lwrl)[1] );
	__m128d phase_mdl_mid = _mm_set_pd( ((double*)&phase_h_mdlr)[0], ((double*)&phase_h_mdll)[1] );

	// print_mm(phase_lwr_mid);
	// print_mm(phase_mdl_mid);

	__m128d phase_sum_hm = _mm_hadd_pd(phase_lwr_mid, phase_mdl_mid);

	// print_mm(phase_sum_hm);
}

static inline void _process(__m128d *phase_h_lwrl_p, __m128d *phase_h_mdll_p, double *phase_lwr, double *phase_mdl, double *res) {



	__m128d phase_h_lwrl = _mm_loadu_pd(phase_lwr);
	// __m128d phase_h_lwrl = *phase_h_lwrl_p;
	__m128d phase_h_lwrr = _mm_loadu_pd(phase_lwr + 2);
	// *phase_h_lwrl_p = phase_h_lwrr;
	__m128d phase_h_mdll = _mm_loadu_pd(phase_mdl);
	// __m128d phase_h_mdll = *phase_h_mdll_p;
	__m128d phase_h_mdlr = _mm_loadu_pd(phase_mdl + 2);
	// *phase_h_mdll_p = phase_h_mdlr;

	// print_mm(phase_h_lwrl);
	// print_mm(phase_h_lwrr);
	// print_mm(phase_h_mdll);
	// print_mm(phase_h_mdlr);


	__m128d phase_sum_vl = _mm_add_pd(phase_h_lwrl, phase_h_mdll);
	__m128d phase_sum_vr = _mm_add_pd(phase_h_lwrr, phase_h_mdlr);

	// print_mm(phase_sum_vl);
	// print_mm(phase_sum_vr);

	__m128d phase_sum_hu = _mm_hadd_pd(phase_h_lwrl, phase_h_lwrr);
	__m128d phase_sum_hd = _mm_hadd_pd(phase_h_mdll, phase_h_mdlr);

	// print_mm(phase_sum_hu);
	// print_mm(phase_sum_hd);

	__m128d phase_lwr_mid = _mm_set_pd( ((double*)&phase_h_lwrr)[0], ((double*)&phase_h_lwrl)[1] );
	__m128d phase_mdl_mid = _mm_set_pd( ((double*)&phase_h_mdlr)[0], ((double*)&phase_h_mdll)[1] );

	// print_mm(phase_lwr_mid);
	// print_mm(phase_mdl_mid);

	__m128d phase_sum_hm = _mm_hadd_pd(phase_lwr_mid, phase_mdl_mid);

	_mm_storeu_pd(res, phase_sum_vl);
	_mm_storeu_pd(res + 2, phase_sum_vr);
	_mm_storeu_pd(res + 4, phase_sum_hu);
	_mm_storeu_pd(res + 6, phase_sum_hd);
	_mm_storeu_pd(res + 8, phase_sum_hm);

	// print_mm(phase_sum_hm);
}

static inline void _process3(__m128d *phase_h_lwrl_p, __m128d *phase_h_mdll_p, double *phase_lwr, double *phase_mdl, double *res) {

	static __m128d phase_h_lwrl;
	phase_h_lwrl = _mm_loadu_pd(phase_lwr);
	// __m128d phase_h_lwrl = *phase_h_lwrl_p;
	__m128d phase_h_lwrr = _mm_loadu_pd(phase_lwr + 2);
	// *phase_h_lwrl_p = phase_h_lwrr;
	static __m128d phase_h_mdll;
	phase_h_mdll = _mm_loadu_pd(phase_mdl);
	// __m128d phase_h_mdll = *phase_h_mdll_p;
	__m128d phase_h_mdlr = _mm_loadu_pd(phase_mdl + 2);
	// *phase_h_mdll_p = phase_h_mdlr;

	// print_mm(phase_h_lwrl);
	// print_mm(phase_h_lwrr);
	// print_mm(phase_h_mdll);
	// print_mm(phase_h_mdlr);


	__m128d phase_sum_vl = _mm_add_pd(phase_h_lwrl, phase_h_mdll);
	__m128d phase_sum_vr = _mm_add_pd(phase_h_lwrr, phase_h_mdlr);

	// print_mm(phase_sum_vl);
	// print_mm(phase_sum_vr);

	__m128d phase_sum_hu = _mm_hadd_pd(phase_h_lwrl, phase_h_lwrr);
	__m128d phase_sum_hd = _mm_hadd_pd(phase_h_mdll, phase_h_mdlr);

	// print_mm(phase_sum_hu);
	// print_mm(phase_sum_hd);

	__m128d phase_lwr_mid = _mm_set_pd( ((double*)&phase_h_lwrr)[0], ((double*)&phase_h_lwrl)[1] );
	__m128d phase_mdl_mid = _mm_set_pd( ((double*)&phase_h_mdlr)[0], ((double*)&phase_h_mdll)[1] );

	// print_mm(phase_lwr_mid);
	// print_mm(phase_mdl_mid);

	__m128d phase_sum_hm = _mm_hadd_pd(phase_lwr_mid, phase_mdl_mid);


	phase_h_lwrl = phase_h_lwrr;
	phase_h_mdll = phase_h_mdlr;

	// res[0] = phase_sum_vl
	_mm_storeu_pd(res, phase_sum_vl);
	_mm_storeu_pd(res + 2, phase_sum_vr);
	_mm_storeu_pd(res + 4, phase_sum_hu);
	_mm_storeu_pd(res + 6, phase_sum_hd);
	_mm_storeu_pd(res + 8, phase_sum_hm);
	// print_mm(phase_sum_hm);
}

void _process_n(double *phase_lwr, double *phase_mdl, double *res) {
	for (int i = 0; i < 4; i++) {
		res[i] = phase_lwr[i] + phase_mdl[i];
	}
	for (int i = 0; i < 3; i++) {
		res[4 + i] = phase_lwr[i] + phase_lwr[i + 1];
		res[4 + i + 3] = phase_mdl[i] + phase_mdl[i + 1];
	}
}

void process(double *arr, int Nx, int Ny) {
	double *lwr = arr;
	double *mdl = arr + Ny;
	__m128d phase_h_lwrl = _mm_loadu_pd(lwr);
	__m128d phase_h_mdll = _mm_loadu_pd(mdl);
	double res[10] = { 0 };
	double a = 0;
	for (int i = 0; i < Ny; i += 2) {
		_process(&phase_h_lwrl, &phase_h_mdll, lwr + i, mdl + i, res);
		// _process_n(lwr + i, mdl + i, res);
		a += res[1];
	}
	printf("%lf\n", a);
}

int main() {
	int size = 1 << 27;
	int Nx = 2;
	int Ny = size / 2;
	double *arr = (double*)malloc(sizeof(double) * size);
	for (int i = 0; i < size; i++) {
		arr[i] = i + 1.5;
	}
	// print_arr(arr, Nx, Ny);

	double *phase_lwr = arr + 0 * Ny;
	double *phase_mdl = arr + 1 * Ny;

	process(arr, Nx, Ny);

	/*__m128d phase_h_lwrl = _mm_loadu_pd(phase_lwr);
	__m128d phase_h_lwrr = _mm_loadu_pd(phase_lwr + 2);
	__m128d phase_h_mdll = _mm_loadu_pd(phase_mdl);
	__m128d phase_h_mdlr = _mm_loadu_pd(phase_mdl + 2);

	print_mm(phase_h_lwrl);
	print_mm(phase_h_lwrr);
	print_mm(phase_h_mdll);
	print_mm(phase_h_mdlr);


	__m128d phase_sum_vl = _mm_add_pd(phase_h_lwrl, phase_h_mdll);
	__m128d phase_sum_vr = _mm_add_pd(phase_h_lwrr, phase_h_mdlr);

	print_mm(phase_sum_vl);
	print_mm(phase_sum_vr);

	__m128d phase_sum_hu = _mm_hadd_pd(phase_h_lwrl, phase_h_lwrr);
	__m128d phase_sum_hd = _mm_hadd_pd(phase_h_mdll, phase_h_mdlr);

	print_mm(phase_sum_hu);
	print_mm(phase_sum_hd);

	__m128d phase_lwr_mid = _mm_set_pd( ((double*)&phase_h_lwrr)[0], ((double*)&phase_h_lwrl)[1] );
	__m128d phase_mdl_mid = _mm_set_pd( ((double*)&phase_h_mdlr)[0], ((double*)&phase_h_mdll)[1] );

	print_mm(phase_lwr_mid);
	print_mm(phase_mdl_mid);

	__m128d phase_sum_mid = _mm_hadd_pd(phase_lwr_mid, phase_mdl_mid);

	__m128d phase_sum_hm = _mm_hadd_pd(phase_lwr_mid, phase_mdl_mid);

	print_mm(phase_sum_hm);*/
	

	return 0;
}