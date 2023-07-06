#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include <string.h>

#include <emmintrin.h>
#include <immintrin.h>


// #define DBG
#ifdef DBG
#define RANDARR
#define DBGITER
#define DBGPRINT
#endif

#define I(a, b) ( (a) * Ny + (b) )
#define IV(a, b) ( (a) * Ny / 2 + (b) )

typedef struct {
	double *curr;
	double *next;
	double *phase;
	double *processed_phase;
	int Nx;
	int Ny;
	int Sx;
	int Sy;
} modeling_plane;

void printv(__m128d a) {
	printf("%f %f\n", ((double*)&a)[0], ((double*)&a)[1]);
}

void print_arr(double *arr, int size) {
	for (int i = 0; i < size; i++) {
		printf("%f ", arr[i]);
	}
}

void print_arrv(__m128d *arr, int size) {
	for (int i = 0; i < size; i++) {
		printf("%f %f ", ((double*)&arr[i])[0], ((double*)&arr[i])[1]);
	}
}

int write_to_file(char *filename, double *arr, int size) {
	int flags = O_WRONLY | O_CREAT | O_TRUNC;
	mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH;
	int fd = open(filename, flags, mode);
	if (fd == -1) {
		perror("open");
		return -1;
	}
	if (write(fd, arr, size * sizeof(double)) == -1) {
		perror("write");
		close(fd);
		return -2;
	}
	close(fd);
	return 0;
}


int init_modeling_plane(modeling_plane *plane, int Nx, int Ny, int Sx, int Sy) {

	int alignment = 1 << 4;
	/*if (Ny % alignment != 0) {
		Ny = (Ny / alignment + 1) * alignment;
	}
	if (Ny % alignment != 0) {
		printf("error alignment. exiting\n");
		return -1;
	}*/
	if (Ny % 2 != 0) {
		Ny++;
	}

	plane->Nx = Nx;
	plane->Ny = Ny;
	plane->Sx = Sx;
	plane->Sy = Sy;

	plane->processed_phase = (double*)aligned_alloc(alignment, Ny * sizeof(double));
	plane->curr = (double*)aligned_alloc(alignment, Nx * Ny * sizeof(double));
	plane->next = (double*)aligned_alloc(alignment, Nx * Ny * sizeof(double));
	plane->phase = (double*)aligned_alloc(alignment, Nx * Ny * sizeof(double));
	if (plane->curr == NULL || plane->next == NULL || plane->phase == NULL) {
		perror("aligned_alloc");
		return -1;
	}
	// maybe i can use memset?
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			plane->curr[I(i,j)] = 0.0;
			plane->next[I(i,j)] = 0.0;
			plane->phase[I(i,j)] = 0.01;
		}
	}
	for (int i = 0; i < Nx; i++) {
		if ( (i / (Nx / 5)) % 2 == 0 ) {
			for (int j = 0; j < Ny; j++) {
				plane->phase[I(i,j)] = 0.02;
			}
		}
	}

#ifdef RANDARR
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			plane->curr[I(i,j)] = (double)(rand() % 10 + 1) / 2;
			plane->next[I(i,j)] = (double)(rand() % 10 + 1) / 2;
			plane->phase[I(i,j)] = (double)(rand() % 10 + 1) / 2;
		}
	}
	for (int i = 0; i < Nx; i++) {
		if ( (i / (Nx / 5)) % 2 == 0 ) {
			for (int j = 0; j < Ny; j++) {
				plane->phase[I(i,j)] = (double)(rand() % 10 + 1) / 2;
			}
		}
	}
#endif
	return 0;
}

int destroy_modeling_plane(modeling_plane *plane) {
	return 0;
}

double f(int n, double tou) {
	double rev_gammasq = 1 / 16.0;
	double tmp = (2 * M_PI * (n * tou - 1.5));
	double exp_arg = - ( tmp * tmp * rev_gammasq);
	return exp(exp_arg) * sin(tmp) * 0.5;
}

double line_get_max(double *line, int Ny) {
	double max = line[0];
	for (int i = 0; i < Ny; i++) {
		if (max < line[i] || line[i] < -max) {
			max = line[i];
		}
	}
	return max;
}

void print_m(double *arr, int Nx, int Ny) {
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			printf("%f ", arr[i * Ny + j]);
		}
		printf("\n");
	}
	printf("\n");
}


// ---------------check process dot---------------
// insert before and after vectorized cycle with proper values
void process_dot(int i, int j, double *phase, double *curr, double *next, double phix, double phiy, int Nx, int Ny, double next_mdl_j) {
	// double hu, hd, vl, vr;
	

	// hu = phase_lwr[j - 1] + phase_lwr[j];
	double hu = phase[(i - 1) * Ny + j - 1] + phase[(i - 1) * Ny + j];
	// hd = phase_mdl[j - 1] + phase_mdl[j];
	double hd = phase[i * Ny + j - 1] + phase[i * Ny + j];
	// vl = phase_lwr[j - 1] + phase_mdl[j - 1];
	double vl = phase[(i - 1) * Ny + j - 1] + phase[i * Ny + j - 1];
	// vr = phase_lwr[j] + phase_mdl[j];
	double vr = phase[(i - 1) * Ny + j] + phase[i * Ny + j];


	double curr_l = curr[I(i, j - 1)];
	double curr_m = curr[I(i, j)];
	double curr_r = curr[I(i, j + 1)];
	double curr_u = curr[I(i - 1, j)];
	double curr_d = curr[I(i + 1, j)];

	double elem_x = (curr_r - curr_m) * vr + (curr_l - curr_m) * vl;
	double elem_y = (curr_d - curr_m) * hd + (curr_u - curr_m) * hu;
#ifdef DBGPRINT
	printf("elem_x: %f\n", elem_x);
	printf("elem_y: %f\n", elem_y);
#endif
	elem_x *= phix;
	elem_y *= phiy;

	double tmp = 2.0 * curr_m - next[I(i, j)] + elem_x + elem_y;
	next[I(i, j)] = 2.0 * curr_m - next[I(i, j)] + elem_x + elem_y;
	/*if (next_mdl_j != tmp) {
		printf("(%d, %d): next_mdl_j: %.15f, tmp: %.15f\n", i, j, next_mdl_j, tmp);
	}*/
}

double calc_step(modeling_plane *plane, double tou) {
	double *curr = plane->curr;
	double *next = plane->next;
	double *phase = plane->phase;
	double *pphase = plane->processed_phase;
	int Nx = plane->Nx;
	int Ny = plane->Ny;
	int Sx = plane->Sx;
	int Sy = plane->Sy;

	__m128d *vec_curr = (__m128d*)curr;
	__m128d *vec_next = (__m128d*)next;
	__m128d *vec_phase = (__m128d*)phase;
	__m128d *vec_pphase = (__m128d*)pphase;

	static int n = 1;

	double tousq = tou * tou;
	double hy = 4.0 / (double)(Nx - 1);
	double hx = 4.0 / (double)(Ny - 1);

	__m128d vec_tou = _mm_set1_pd(tou);
	__m128d vec_tousq = _mm_mul_pd(vec_tou, vec_tou);
	__m128d vec_hy = _mm_div_pd(_mm_set1_pd(4.0), _mm_set1_pd((double)(Nx - 1)));
	__m128d vec_hx = _mm_div_pd(_mm_set1_pd(4.0), _mm_set1_pd((double)(Ny - 1)));

	double phixt = tou / hx;
	double phix = phixt * phixt * 0.5;
	double phiyt = tou / hy;
	double phiy = phiyt * phiyt * 0.5;

	__m128d vec_phixt = _mm_div_pd(vec_tou, vec_hx);
	__m128d vec_phix = _mm_mul_pd(vec_phixt, vec_phixt);
	vec_phix = _mm_mul_pd(vec_phix, _mm_set1_pd(0.5));
	__m128d vec_phiyt = _mm_div_pd(vec_tou, vec_hy);
	__m128d vec_phiy = _mm_mul_pd(vec_phiyt, vec_phiyt);
	vec_phiy = _mm_mul_pd(vec_phiy, _mm_set1_pd(0.5));

	int i = 1;
	int j = 1;

	double elem_x = 0.0;
	double elem_y = 0.0;

	__m128d vec_elem_x = _mm_setzero_pd();
	__m128d vec_elem_y = _mm_setzero_pd();

	double *curr_lwr = NULL;
	double *curr_mdl = NULL;
	double *curr_upr = NULL;

	__m128d *vec_curr_lwr = (__m128d*)curr_lwr;
	__m128d *vec_curr_mdl = (__m128d*)curr_mdl;
	__m128d *vec_curr_upr = (__m128d*)curr_upr;

	double *phase_lwr = NULL;
	double *phase_mdl = NULL;

	__m128d *vec_phase_lwr = (__m128d*)phase_lwr;
	__m128d *vec_phase_mdl = (__m128d*)phase_mdl;

	double *next_mdl = NULL;

	__m128d *vec_next_mdl = (__m128d*)next_mdl;

	double hu, hd, vl, vr;

	__m128d vec_hu, vec_hd, vec_vl, vec_vr;


	double max = curr[0];





	

	// print_arr(pphase, Ny);
	// printf("\n");
#ifdef DBGPRINT
	printf("phase\n");
	print_m(phase, Nx, Ny);
	printf("\n\n");
	printf("curr\n");
	print_m(curr, Nx, Ny);
	printf("\n");
#endif
	int k = 0;

	for (i = 1; i < Nx - 1; i++) {

		curr_lwr = curr + (i - 1) * Ny;
		curr_mdl = curr + (i) * Ny;
		curr_upr = curr + (i + 1) * Ny;

		phase_lwr = phase + (i - 1) * Ny;
		phase_mdl = phase + (i) * Ny;

		next_mdl = next + (i) * Ny;

		vec_curr_lwr = vec_curr + (i - 1) * Ny / 2;
		vec_curr_mdl = vec_curr + (i) * Ny / 2;
		vec_curr_upr = vec_curr + (i + 1) * Ny / 2;

		vec_phase_lwr = vec_phase + (i - 1) * Ny / 2;
		vec_phase_mdl = vec_phase + (i) * Ny / 2;

		vec_next_mdl = vec_next + (i) * Ny / 2;

		process_dot(i, 1, phase, curr, next, phix, phiy, Nx, Ny, 0); // remove last argument

		// for (j = 1; j < Ny - 1; j++) {
		for (k = 1; k < Ny / 2 - 1; k++) {

			__m128d vec_phase_ul = vec_phase[IV(i - 1, k - 1)];
			__m128d vec_phase_ur = vec_phase[IV(i - 1, k)];
			__m128d vec_phase_dl = vec_phase[IV(i, k - 1)];
			__m128d vec_phase_dr = vec_phase[IV(i, k)];

			__m128d vec_curr_u = vec_curr[IV(i - 1, k)];
			__m128d vec_curr_d = vec_curr[IV(i + 1, k)];
			__m128d vec_curr_l = vec_curr[IV(i, k - 1)];
			__m128d vec_curr_r = vec_curr[IV(i, k + 1)];
			__m128d vec_curr_m = vec_curr[IV(i, k)];

#ifdef DBGPRINT
			printf("iter: (%d, %d) (i, k)\n", i, k);
			printv(vec_phase_ul);
			printv(vec_phase_ur);
			printv(vec_phase_dl);
			printv(vec_phase_dr);
			printf("\n");
			printv(vec_curr_u);
			printv(vec_curr_d);
			printv(vec_curr_l);
			printv(vec_curr_r);
			printv(vec_curr_m);
			printf("\n");
#endif

			// phase surrogats
			__m128d vec_phase_u_s = _mm_shuffle_pd(vec_phase_ul, vec_phase_ur, 1);
			__m128d vec_phase_d_s = _mm_shuffle_pd(vec_phase_dl, vec_phase_dr, 1);

			// curr surrogats
			__m128d vec_curr_l_s = _mm_shuffle_pd(vec_curr_l, vec_curr_m, 1);
			__m128d vec_curr_r_s = _mm_shuffle_pd(vec_curr_m, vec_curr_r, 1);
#ifdef DBGPRINT
			printv(vec_phase_u_s);
			printv(vec_phase_d_s);
			printv(vec_curr_l_s);
			printv(vec_curr_r_s);
			printf("\n");
#endif
			vec_vl = vec_phase_u_s + vec_phase_d_s;
			vec_vr = vec_phase_ur + vec_phase_dr;
			vec_hu = vec_phase_u_s + vec_phase_ur;
			vec_hd = vec_phase_d_s + vec_phase_dr;

#ifdef DBGPRINT
			// check values
			printv(vec_vl);
			printv(vec_vr);
			printv(vec_hu);
			printv(vec_hd);
			printf("\n");
#endif
			vec_elem_x = (vec_curr_r_s - vec_curr_m) * vec_vr + (vec_curr_l_s - vec_curr_m) * vec_vl;
			vec_elem_y = (vec_curr_d - vec_curr_m) * vec_hd + (vec_curr_u - vec_curr_m) * vec_hu;
#ifdef DBGPRINT
			printv(vec_elem_x);
			printv(vec_elem_y);

			printf("\n\n");
#endif
			vec_elem_y *= vec_phiy;
			vec_elem_x *= vec_phix;

			vec_next_mdl[k] = _mm_set1_pd(2.0) * vec_curr_m - vec_next_mdl[k] + vec_elem_x + vec_elem_y;

			/*if (max < next_mdl[j] || next_mdl[j] < -max) {
				max = next_mdl[j];
			}*/
		}
		
		process_dot(i, Ny - 2, phase, curr, next, phix, phiy, Nx, Ny, 0); // remove last argument
	}
	

	next[I(Sx, Sy)] += tousq * f(n, tou);



	// print_m(phase, Nx, Ny);
	// print_arr(phase + (Nx - 1) * Ny, Ny);
	// printf("\n");
	plane->curr = next;
	plane->next = curr;

	n++;

	// printf("%.10f\n", max);
#ifdef DBGPRINT
	print_m(phase, Nx, Ny);
	print_m(plane->curr, Nx, Ny);
#endif

	return max;
}

/*void print_m(double *arr, int Nx, int Ny) {
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			if (arr[I(i,j)] > 1000 || arr[I(i,j)] < -1000) {
				printf("(%d, %d) %f\n", i, j, arr[I(i,j)]);
			}
			
		}
	}
}*/

void print_csv(double *arr, int Nx, int Ny) {
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			printf("%f;", arr[I(i,j)]);
		}
		printf("\b\n");
	}
}

int main(int argc, char *argv[]) {
	// double tou = 0.01;
	int Nx = 0, Ny = 0, Nt = 0;
	int opt = 0;
	int k = 0;
	while ( (opt = getopt(argc, argv, "x:y:t:k:")) != -1 ) {
		switch (opt) {
			case 'x':
				Nx = atoi(optarg);
				break;
			case 'y':
				Ny = atoi(optarg);
				break;
			case 't':
				Nt = atoi(optarg);
				break;
			case 'k':
				k = atoi(optarg);
				break;
			case '?':
				printf("error: no such arg\n");
				break;
		}
	}
	printf("Nx: %d, Ny: %d, Nt: %d\n", Nx, Ny, Nt);

	/*int Sx = Nx / 2;
	int Sy = Ny / 2;*/

	int Sx = 1;
	int Sy = 1;

	/*int Sx = Nx - 2;
	int Sy = Ny - 2;*/
	
	modeling_plane plane;
	if (init_modeling_plane(&plane, Nx, Ny, Sx, Sy) == -1) {
		fprintf(stderr, "error in init_modeling_plane\n");
		exit(-1);
	}

	double tou = 0.01;
	int iters = (double)Nt / tou;
	printf("Nt: %d, tou: %f, iters: %d\n", Nt, tou, iters);

	char fname[100] = { 0 };

	// --------dbg-------
#ifdef DBGITER
	iters = 2;
#endif
	// --------dbg-------

	for (int i = 1; i < iters; i++) {
		calc_step(&plane, tou);
		sprintf(fname, "data/vcurr%d", i);
		if (i % 50 == 0) {
			write_to_file(fname, plane.curr, Nx * Ny);
		}
		// write_to_file(fname, plane.curr, Nx * Ny);
		if (i == k) {
			print_csv(plane.curr, Nx, Ny);
		}
	}

	return 0;
}