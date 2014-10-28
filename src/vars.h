#ifndef __VARS_H__ 
#define __VARS_H__ 

#include <map>
#include <string>

struct var {
	std::string name;
	enum t { t_int, t_float };
	t type;
	union {
		int int_val;
		float float_val;
	};
};

extern std::map<std::string, var> vars;

template<typename T> struct var_type_map {};
template<> struct var_type_map<int> { static const var::t type = var::t_int; };
template<> struct var_type_map<float> { static const var::t type = var::t_float; };

template<typename T> inline void declare_variable(const std::string &s, T val = T()) {
	var v;
	v.name = s;
	v.type = var_type_map<T>::type;
	if (v.type == var::t_int) v.int_val = val;
	else                    v.float_val = val;
	vars[v.name] = v;
}

#endif

