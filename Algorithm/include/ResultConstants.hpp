/**
 * @file ResultConstants.hpp
 * @author Michal Solanik
 * @brief Constants needed for analysis of log files for all models.
 * @version 0.2
 * @date 2022-10-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef RESULTS_CONTANTS
#define RESULTS_CONTANTS

/**
 * @brief Proton mass.
 * 
 */
static double m0 = 1.67261e-27;

/**
 * @brief The fundamental electrical charge.
 * 
 */
static double q = 1.60219e-19;

/**
 * @brief Speed of light.
 * 
 */
static double c = 2.99793e8;

/**
 * @brief Rest energy in [J].
 * 
 */
static double T0w = m0 * c * c;

/**
 * @brief Rest energy in [GeV].
 * 
 */
static double T0 = m0 * c * c / (q * 1e9);

/**
 * @brief Constant for calculating log bins. 
 * 
 */
static int NT = 80;

/**
 * @brief Constant for calculating log bins. 
 * 
 */
static double Tmin = 0.01;

/**
 * @brief Constant for calculating log bins.  
 * 
 */
static double Tmax = 200.0;

/**
 * @brief Constant for calculating log bins.  
 * 
 */
static double dlT = log10(Tmax / Tmin) / NT;

/**
 * @brief Constant for calculating log bins.  
 * 
 */
static double X = log10(Tmax);

/**
 * @brief Constant for calculating log bins.  
 * 
 */
static double larg = exp((dlT / 2.0) * log(10.0)) - exp((-1.0 * dlT / 2.0) * log(10.0));

/**
 * @brief Constant for calculating log bins.  
 * 
 */
const static double Tr = 0.938;

/**
 * @brief Array of bins for 2D models.
 * 
 */
const double SPbins[32] = { 0, 0.01, 0.015, 0.0225, 0.03375, 0.050625,
	0.0759375, 0.113906, 0.170859, 0.256289, 0.384434, 0.57665,
	0.864976, 1.29746, 1.9462, 2.91929, 4.37894, 6.56841, 9.85261,
	14.7789, 22.1684, 33.2526, 49.8789, 74.8183, 112.227, 168.341,
	252.512, 378.768, 568.151, 852.227, 1278.34};

/**
 * @brief Array of bins for 2D models.
 * 
 */
const double UlyssesBins[4] = { 0, 0.125, 0.250, 2.0 };

/**
 * @brief Array of bins for 2D models.
 * 
 */
const double AmsBins[72] = { 4.924000e-01, 6.207000e-01, 7.637000e-01, 9.252000e-01, 1.105000e+00, 
	1.303000e+00, 1.523000e+00, 1.765000e+00, 2.034000e+00, 2.329000e+00, 2.652000e+00, 3.005000e+00, 
	3.390000e+00, 3.810000e+00, 4.272000e+00, 4.774000e+00, 5.317000e+00, 5.906000e+00, 6.546000e+00, 
	7.236000e+00, 7.981000e+00, 8.787000e+00, 9.653000e+00, 1.060000e+01, 1.160000e+01, 1.264000e+01, 
	1.379000e+01, 1.504000e+01, 1.639000e+01, 1.784000e+01, 1.938000e+01, 2.103000e+01, 2.283000e+01, 
	2.478000e+01, 2.683000e+01, 2.903000e+01, 3.138000e+01, 3.387000e+01, 3.657000e+01, 3.947000e+01, 
	4.257000e+01, 4.587000e+01, 4.942000e+01, 5.322000e+01, 5.727000e+01, 6.162000e+01, 6.632000e+01, 
	7.137000e+01, 7.677000e+01, 8.257000e+01, 8.882000e+01, 9.557000e+01, 1.031000e+02, 1.111000e+02, 
	1.196000e+02, 1.291000e+02, 1.401000e+02, 1.526000e+02, 1.666000e+02, 1.826000e+02, 2.006000e+02, 
	2.211000e+02, 2.451000e+02, 2.741000e+02, 3.096000e+02, 3.536000e+02, 4.091000e+02, 4.821000e+02, 
	5.831000e+02, 7.316000e+02, 9.751000e+02, 1.464000e+03 };

#endif