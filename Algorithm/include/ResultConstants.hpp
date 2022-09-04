#ifndef RESULTS_CONTANTS
#define RESULTS_CONTANTS

static double m0 = 1.67261e-27;
static double q = 1.60219e-19;
static double c = 2.99793e8;
static double T0w = m0 * c * c;
static double T0 = m0 * c * c / (q * 1e9);
static int NT = 80;
static double Tmin = 0.01;
static double Tmax = 200.0;
static double dlT = log10(Tmax / Tmin) / NT;
static double X = log10(Tmax);
static double larg = exp((dlT / 2.0) * log(10.0)) - exp((-1.0 * dlT / 2.0) * log(10.0));
const static double Tr = 0.938;
const double SPbins[31] = { 0, 0.01, 0.015, 0.0225, 0.03375, 0.050625,
	0.0759375, 0.113906, 0.170859, 0.256289, 0.384434, 0.57665,
	0.864976, 1.29746, 1.9462, 2.91929, 4.37894, 6.56841, 9.85261,
	14.7789, 22.1684, 33.2526, 49.8789, 74.8183, 112.227, 168.341,
	252.512, 378.768, 568.151, 852.227, 1278.34};

#endif