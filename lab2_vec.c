#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include <emmintrin.h>
#include <immintrin.h>

#define I(a, b) ( (a) * Ny + (b) )

#define CALC_ELEMS(j, elem_x, elem_y, vl, vr, hu, hm)                                                     \
{                                                                                                         \
	(elem_x) = (curr_mdl[(j) + 1] - curr_mdl[(j)]) * (vr1) + (curr_mdl[(j) - 1] - curr_mdl[(j)]) * (vl1); \
	(elem_y) = (curr_upr[(j)    ] - curr_mdl[(j)]) * (hm1) + (curr_lwr[(j)    ] - curr_mdl[(j)]) * (hu1); \
}

#define CALC_LINE(i, curr, next)                                                       \
{                                                                                      \
	curr_lwr = (curr) + ((i) - 1) * Ny;                                                \
	curr_mdl = (curr) + (i) * Ny;                                                      \
	curr_upr = (curr) + ((i) + 1) * Ny;                                                \
                                                                                       \
	phase_lwr = (phase) + ((i) - 1) * Ny;                                              \
	phase_mdl = (phase) + (i) * Ny;                                                    \
                                                                                       \
	next_mdl = (next) + (i) * Ny;                                                      \
                                                                                       \
	for (j = 1; j < Ny - 1; j += 3) {                                                  \
                                                                                       \
		double res[10];                                                                \
                                                                                       \
		_process(NULL,                                                                 \
				 NULL,                                                                 \
				 phase_lwr + j - 1,                                                    \
				 phase_mdl + j - 1,                                                    \
				 res);                                                                 \
                                                                                       \
		double vl1 = res[0];                                                           \
		double vr1 = res[1];                                                           \
		double vl3 = res[2];                                                           \
		double vr3 = res[3];                                                           \
		double vl2 = vr1;                                                              \
		double vr2 = vl3;                                                              \
                                                                                       \
		double hu1 = res[4];                                                           \
		double hu3 = res[5];                                                           \
		double hm1 = res[6];                                                           \
		double hm3 = res[7];                                                           \
                                                                                       \
		double hu2 = res[8];                                                           \
		double hm2 = res[9];                                                           \
                                                                                       \
                                                                                       \
		double elem_x1 = 0.0;                                                          \
		double elem_y1 = 0.0;                                                          \
                                                                                       \
		CALC_ELEMS(j, elem_x1, elem_y1, vl1, vr1, hu1, hm1);                           \
                                                                                       \
		double elem_x2 = 0.0;                                                          \
		double elem_y2 = 0.0;                                                          \
                                                                                       \
		CALC_ELEMS(j + 1, elem_x2, elem_y2, vl2, vr2, hu2, hm2);                       \
                                                                                       \
		double elem_x3 = 0.0;                                                          \
		double elem_y3 = 0.0;                                                          \
                                                                                       \
		CALC_ELEMS(j + 2, elem_x3, elem_y3, vl3, vr3, hu3, hm3);                       \
			                                                                           \
                                                                                       \
		elem_x1 *= phix;                                                               \
		elem_y1 *= phiy;                                                               \
                                                                                       \
		elem_x2 *= phix;                                                               \
		elem_y2 *= phiy;                                                               \
                                                                                       \
		elem_x3 *= phix;                                                               \
		elem_y3 *= phiy;                                                               \
                                                                                       \
		next_mdl[j    ] = 2.0 * curr_mdl[j    ] - next_mdl[j    ] + elem_x1 + elem_y1; \
		next_mdl[j + 1] = 2.0 * curr_mdl[j + 1] - next_mdl[j + 1] + elem_x2 + elem_y2; \
		next_mdl[j + 2] = 2.0 * curr_mdl[j + 2] - next_mdl[j + 2] + elem_x3 + elem_y3; \
	}                                                                                  \
}

typedef struct {
	double *curr;
	double *next;
	double *phase;
	int Nx;
	int Ny;
	int Sx;
	int Sy;
} modeling_plane;

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
	plane->Nx = Nx;
	plane->Ny = Ny;
	plane->Sx = Sx;
	plane->Sy = Sy;
	plane->curr = (double*)malloc(Nx * Ny * sizeof(double));
	plane->next = (double*)malloc(Nx * Ny * sizeof(double));
	plane->phase = (double*)malloc(Nx * Ny * sizeof(double));
	if (plane->curr == NULL || plane->next == NULL || plane->phase == NULL) {
		perror("malloc");
		return -1;
	}
	// maybe i can use memset?
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			// plane->prev[I(i,j)] = 0.0;
			plane->curr[I(i,j)] = 0.0;
			plane->next[I(i,j)] = 0.0;
			plane->phase[I(i,j)] = 0.01;
		}
	}
	for (int i = Nx / 2; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			plane->phase[I(i,j)] = 0.02;
		}
	}
	return 0;
}

double f(int n, double tou) {
	double rev_gammasq = 1 / 16.0;
	double tmp = (2 * M_PI * (n * tou - 1.5));
	double exp_arg = - ( tmp * tmp * rev_gammasq);
	return exp(exp_arg) * sin(tmp) * 0.5;
}

static inline void _process(__m128d *phase_h_lwrl_p, __m128d *phase_h_mdll_p, double *phase_lwr, double *phase_mdl, double *res) {



	__m128d phase_h_lwrl = _mm_loadu_pd(phase_lwr);
	__m128d phase_h_lwrr = _mm_loadu_pd(phase_lwr + 2);
	__m128d phase_h_mdll = _mm_loadu_pd(phase_mdl);
	__m128d phase_h_mdlr = _mm_loadu_pd(phase_mdl + 2);


	__m128d phase_sum_vl = _mm_add_pd(phase_h_lwrl, phase_h_mdll);
	__m128d phase_sum_vr = _mm_add_pd(phase_h_lwrr, phase_h_mdlr);

	__m128d phase_sum_hu = _mm_hadd_pd(phase_h_lwrl, phase_h_lwrr);
	__m128d phase_sum_hd = _mm_hadd_pd(phase_h_mdll, phase_h_mdlr);

	__m128d phase_lwr_mid = _mm_set_pd( ((double*)&phase_h_lwrr)[0], ((double*)&phase_h_lwrl)[1] );
	__m128d phase_mdl_mid = _mm_set_pd( ((double*)&phase_h_mdlr)[0], ((double*)&phase_h_mdll)[1] );

	__m128d phase_sum_hm = _mm_hadd_pd(phase_lwr_mid, phase_mdl_mid);

	_mm_storeu_pd(res, phase_sum_vl);
	_mm_storeu_pd(res + 2, phase_sum_vr);
	_mm_storeu_pd(res + 4, phase_sum_hu);
	_mm_storeu_pd(res + 6, phase_sum_hd);
	_mm_storeu_pd(res + 8, phase_sum_hm);
}


/*
verticals / horizontals

     j->
  *--*--*--*--> upper
  |  |  |  |
i *--*--*--*--> middle
| |  |
v |  v
  v  right
left
*/

void calc_step(modeling_plane *plane, double tou) {
	// double *prev = plane->prev;
	double *curr = plane->curr;
	double *next = plane->next;
	double *phase = plane->phase;
	int Nx = plane->Nx;
	int Ny = plane->Ny;
	int Sx = plane->Sx;
	int Sy = plane->Sy;

	static int n = 1;

	double tousq = tou * tou;
	double hy = 4.0 / (double)(Nx - 1);
	double hx = 4.0 / (double)(Ny - 1);
	
	double phixt = tou / hx;
	double phix = phixt * phixt * 0.5;
	double phiyt = tou / hy;
	double phiy = phiyt * phiyt * 0.5;

	int i = 1;
	int j = 1;

	double *curr_lwr = NULL;
	double *curr_mdl = NULL;
	double *curr_upr = NULL;

	double *phase_lwr = NULL;
	double *phase_mdl = NULL;

	double *prev_mdl = NULL;
	double *next_mdl = NULL;

	for (i = 1; i < Nx - 1; i++) {

		CALC_LINE(i, curr, next);
		
	}
	
	next[I(Sx, Sy)] += tousq * f(n, tou);

	plane->curr = next;
	plane->next = curr;
	n++;
}

void print_m(double *arr, int Nx, int Ny) {
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			if (arr[I(i,j)] > 1000 || arr[I(i,j)] < -1000) {
				printf("(%d, %d) %f\n", i, j, arr[I(i,j)]);
			}
			
		}
	}
}

void print_csv(double *arr, int Nx, int Ny) {
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			printf("%f;", arr[I(i,j)]);
		}
		printf("\b\n");
	}
}

int main(int argc, char *argv[]) {
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

	int Sx = Nx / 2;
	int Sy = Ny / 2;
	
	modeling_plane plane;
	if (init_modeling_plane(&plane, Nx, Ny, Sx, Sy) == -1) {
		fprintf(stderr, "error in init_modeling_plane\n");
		exit(-1);
	}

	double tou = 0.01;
	int iters = (double)Nt / tou;
	printf("Nt: %d, tou: %f, iters: %d\n", Nt, tou, iters);

	char fname[100] = { 0 };

	for (int i = 1; i < iters; i+=1) {
		calc_step(&plane, tou);

		sprintf(fname, "data/acurr%d", i);
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