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

typedef struct {
	double *prev;
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
	plane->prev = (double*)malloc(Nx * Ny * sizeof(double));
	plane->curr = (double*)malloc(Nx * Ny * sizeof(double));
	// plane->next = (double*)malloc(Nx * Ny * sizeof(double));
	plane->next = plane->prev;
	plane->phase = (double*)malloc(Nx * Ny * sizeof(double));
	if (plane->prev == NULL || plane->curr == NULL || plane->next == NULL || plane->phase == NULL) {
		perror("malloc");
		return -1;
	}
	// maybe i can use memset?
	for (int i = 0; i < Nx; i++) {
		for (int j = 0; j < Ny; j++) {
			plane->prev[I(i,j)] = 0.0;
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



void calc_step(modeling_plane *plane, double tou) {
	double *prev = plane->prev;
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

	double elem_x = 0.0;
	double elem_y = 0.0;

	double *curr_lwr = NULL;
	double *curr_mdl = NULL;
	double *curr_upr = NULL;

	double *phase_lwr = NULL;
	double *phase_mdl = NULL;

	double *prev_mdl = NULL;
	double *next_mdl = NULL;
	for (i = 1; i < Nx - 1; i++) {

		curr_lwr = curr + (i - 1) * Ny;
		curr_mdl = curr + (i) * Ny;
		curr_upr = curr + (i + 1) * Ny;

		phase_lwr = phase + (i - 1) * Ny;
		phase_mdl = phase + (i) * Ny;

		next_mdl = next + (i) * Ny;
		prev_mdl = prev + (i) * Ny;

		for (j = 1; j < Ny - 1; j++) {

			elem_x = (curr_mdl[j+1] - curr_mdl[j]) * (phase_lwr[j] + phase_mdl[j]) +
						(curr_mdl[j - 1] - curr_mdl[j]) * (phase_lwr[j - 1] + phase_mdl[j - 1]);
			elem_y = (curr_upr[j] - curr_mdl[j]) * (phase_mdl[j - 1] + phase_mdl[j]) +
						(curr_lwr[j] - curr_mdl[j]) * (phase_lwr[j - 1] + phase_lwr[j]);

			

			elem_x *= phix;
			elem_y *= phiy;

			next_mdl[j] = 2.0 * curr_mdl[j] - prev_mdl[j] + elem_x + elem_y;
			// next[I(i, j)] = 2.0 * curr[I(i, j)] - prev[I(i, j)] + elem_x + elem_y;

		}
	}
	next[I(Sx, Sy)] += tousq * f(n, tou);

	plane->prev = curr;
	plane->curr = next;
	plane->next = prev;
	plane->next = plane->prev;
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

	for (int i = 1; i < iters; i++) {
		calc_step(&plane, tou);
		// sprintf(fname, "prev%d", i);
		// write_to_file(fname, plane.prev, Nx * Ny);
		sprintf(fname, "data/curr%d", i);
		if (i % 50 == 0) {
			write_to_file(fname, plane.curr, Nx * Ny);
		}
		// write_to_file(fname, plane.curr, Nx * Ny);
		/*if (i == k) {
			print_csv(plane.curr, Nx, Ny);
		}*/
		// sprintf(fname, "next%d", i);
		// write_to_file(fname, plane.next, Nx * Ny);
		/*if (i % 10 == 0) {
			write_to_file(fname, plane.curr, Nx * Ny);
		}*/
	}

	return 0;
}