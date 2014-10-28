#ifndef __GI_ALGORITHM_H__ 
#define __GI_ALGORITHM_H__ 

#include <string>
#include <list>
#include <vector>
#include <stdexcept>

class gi_algorithm {
protected:
	static std::vector<gi_algorithm*> algorithms;
	static gi_algorithm *selected;
	bool activated;
	std::string name;

public:
	gi_algorithm(const std::string &name) : activated(false) {
	}
	virtual void activate() {
		activated = true;
	}
	virtual void compute() = 0;

	static void select(const std::string &name) {
		for (int i = 0; i < algorithms.size(); ++i)
			if (algorithms[i]->name == name) {
				selected = algorithms[i];
				if (!selected->activated)
					selected->activate();
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

