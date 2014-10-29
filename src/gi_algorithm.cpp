#include "gi_algorithm.h"


std::vector<gi_algorithm*> gi_algorithm::algorithms;
gi_algorithm *gi_algorithm::selected = 0;
rta::rt_set *gi_algorithm::original_rt_set = 0;


/* vim: set foldmethod=marker: */

