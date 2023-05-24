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

int main() {
	int Nx = 2;
	int Ny = 4;
	double arr[8];
	for (int i = 0; i < 8; i++) {
		arr[i] = i + 0.5;
	}

	double *phase_lwr = arr + 0 * Ny;
	double *phase_mdl = arr + 1 * Ny;

	__m128d phase_h_lwr1 = _mm_loadu_pd(phase_lwr);
	__m128d phase_h_lwr2 = _mm_loadu_pd(phase_lwr + 2);
	__m128d phase_h_mdl1 = _mm_loadu_pd(phase_mdl);
	__m128d phase_h_mdl2 = _mm_loadu_pd(phase_mdl + 2);

	__m128d phase_sum_v1 = _mm_add_pd(phase_h_lwr1, phase_h_mdl1);
	__m128d phase_sum_v2 = _mm_add_pd(phase_h_lwr2, phase_h_mdl2);

	__m128d phase_sum_h1 = _mm_hadd_pd(phase_h_lwr1, phase_h_lwr2);
	__m128d phase_sum_h2 = _mm_hadd_pd(phase_h_mdl1, phase_h_mdl2);

	__m128d phase_lwr_mid = _mm_set_pd( ((double*)&phase_h_lwr1)[1], ((double*)&phase_h_lwr2)[0] );
	__m128d phase_mdl_mid = _mm_set_pd( ((double*)&phase_h_mdl1)[1], ((double*)&phase_h_mdl2)[0] );


	

	return 0;
}