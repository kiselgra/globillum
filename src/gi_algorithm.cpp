#include "gi_algorithm.h"

int init_path_samples = 2, init_path_length = 2, init_light_samples = 2;

std::vector<gi_algorithm*> gi_algorithm::algorithms;
gi_algorithm *gi_algorithm::selected = 0;
rta::rt_set *gi_algorithm::original_rt_set = 0;
rta::rt_set *gi_algorithm::original_subd_set = 0;
texture_ref gi_algorithm::result = {-1};


/* vim: set foldmethod=marker: */

