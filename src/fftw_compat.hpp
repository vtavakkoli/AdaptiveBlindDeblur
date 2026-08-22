#pragma once

#if __has_include(<fftw3.h>)
#include <fftw3.h>
#else
extern "C" {
typedef double fftw_complex[2];
typedef struct fftw_plan_s* fftw_plan;
void* fftw_malloc(unsigned long n);
void fftw_free(void* p);
fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex* out, unsigned flags);
fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex* in, double* out, unsigned flags);
void fftw_execute(const fftw_plan p);
void fftw_destroy_plan(fftw_plan p);
int fftw_init_threads(void);
void fftw_plan_with_nthreads(int nthreads);
}
#ifndef FFTW_ESTIMATE
#define FFTW_ESTIMATE (1U << 6)
#endif
#endif
