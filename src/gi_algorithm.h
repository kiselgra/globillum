#ifndef __GI_ALGORITHM_H__ 
#define __GI_ALGORITHM_H__ 

#include <libhyb/rta-cgls-connection.h>

#include <string>
#include <list>
#include <vector>
#include <stdexcept>
#include <iostream>

class gi_algorithm {
public:
	static rta::rt_set *original_rt_set;
	static gi_algorithm *selected;
protected:
	static std::vector<gi_algorithm*> algorithms;
	bool activated;
	std::string name;

public:
	gi_algorithm(const std::string &name) : activated(false), name(name) {
		for (auto *a : algorithms)
			if (a->name == name)
				throw std::logic_error(std::string("the algorithm ") + name + " is already registered.");
		algorithms.push_back(this);
	}
	virtual void activate(rta::rt_set *orig_set) {
		activated = true;
	}
	virtual void compute() = 0;

	static void select(const std::string &name) {
		for (int i = 0; i < algorithms.size(); ++i)
			if (algorithms[i]->name == name) {
				selected = algorithms[i];
				if (!selected->activated)
					selected->activate(original_rt_set);
				return;
			}
		throw std::runtime_error("no gi algorithm called " + name);
	}
	static gi_algorithm* get(const std::string &name) {
		for (int i = 0; i < algorithms.size(); ++i)
			if (algorithms[i]->name == name) {
				return algorithms[i];
			}
		return 0;
	}
	static std::list<std::string> list() {
		std::list<std::string> l;
		for (int i = 0; i < algorithms.size(); ++i)
			l.push_back(algorithms[i]->name);
		return l;
	}

};

#endif

