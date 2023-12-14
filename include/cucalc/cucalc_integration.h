#pragma once

#define TRAPEZOIDAL 1
#define SIMPSON_1_3 2
#define SIMPSON_3_8 3
#define BOOLE 4

double cucalc_integration_trapez(void *func, double a, double b, size_t steps);

double cucalc_integration_simpson_1_3(void *func, double a, double b, size_t steps);

double cucalc_integration_simpson_3_8(void *func, double a, double b, size_t steps);

double cucalc_integration_boole(void *func, double a, double b, size_t steps);
