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

#endif