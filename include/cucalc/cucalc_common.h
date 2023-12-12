#pragma once

void cudaErrorCheck(cudaError_t err, const char* message, bool abort);

double cucalc_reduction_sum(double *d_fx, size_t size);
