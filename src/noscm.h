#ifndef __NOSCM_H__ 
#define __NOSCM_H__ 

#include <libcgls/cgls.h>

#include <map>
#include <string>
#include <stdexcept>

namespace scene {

	class scene {
		static std::map<std::string, scene*> scenes;
		std::string name;
	protected:
		scene(const std::string &name) : name(name) {
			scenes[name] = this;
		}
	public:
		virtual void load() = 0;
		static scene *selected;
		static void select(const std::string &name) {
			std::map<std::string, scene*>::iterator it = scenes.find(name);
			if (it == scenes.end())
				throw std::runtime_error(std::string("scene not found: '") + name + "'");
			selected = it->second;
		}
	};

    drawelement_ref create_drawelement(const char *name, mesh_ref mesh, material_ref mat, vec3f *bbmin, vec3f *bbmax);
    drawelement_ref create_drawelement_idx(const char *name, mesh_ref mesh, material_ref mat, unsigned int pos, unsigned int len, vec3f *bbmin, vec3f *bbmax);
    void setup_scene(const std::string &mainfile, float merge_factor, vec3f &min, vec3f &max);
}

#endif

