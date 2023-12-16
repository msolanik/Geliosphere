#ifndef BATCH_RUN
#define BATCH_RUN

#include "AbstractAlgorithm.hpp"


class BatchRun : public AbstractAlgorithm
{
public:
	void runAlgorithm(ParamsCarrier *singleTone);
};

#endif