#pragma once

double* cucalc_derivative_backward(void *func, double a, double b, size_t steps);

double* cucalc_derivative_forward(void *func, double a, double b, size_t steps);

double* cucalc_derivative_central(void *func, double a, double b, size_t steps);