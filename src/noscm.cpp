#include "noscm.h"

#include "cmdline.h"

#include <libcgls/cgls.h>


using namespace std;

extern scene_ref the_scene;

namespace scene {

    drawelement_ref create_drawelement(const char *name, mesh_ref mesh, material_ref mat, vec3f *bbmin, vec3f *bbmax) {
        shader_ref shader = { -1 };
        drawelement_ref de = make_drawelement(name, mesh, shader, mat);
        prepend_drawelement_uniform_handler(de, (uniform_setter_t)default_matrix_uniform_handler);
        prepend_drawelement_uniform_handler(de, (uniform_setter_t)default_material_uniform_handler);
		scene_add_drawelement(the_scene, de);
        set_drawelement_bounding_box(de, bbmin, bbmax);
        return de;
    }

    drawelement_ref create_drawelement_idx(const char *name, mesh_ref mesh, material_ref mat, unsigned int pos, unsigned int len, vec3f *bbmin, vec3f *bbmax) {
        drawelement_ref de = create_drawelement(name, mesh, mat, bbmin, bbmax);
        set_drawelement_index_buffer_range(de, pos, len);
        return de;
    }

    void setup_scene(const std::string &mainfile, float merge_factor, vec3f &min, vec3f &max) {
        vec4f col = vec4f(1, 0, 0, 1);
        material_ref fallback = make_material((char*)"fallback", &col, &col, &col);
        load_objfile_and_create_objects_with_single_vbo_keeping_cpu_data(mainfile.c_str(), mainfile.c_str(), &min, &max, create_drawelement_idx, fallback, merge_factor);

        float near = 1, far = 1;
        vec3f diam, diam_half, center, pos;
        sub_components_vec3f(&diam, &max, &min);
        mul_vec3f_by_scalar(&diam_half, &diam, 0.5);
        add_components_vec3f(&center, &min, &diam_half);
        float distance = length_of_vec3f(&diam);
        vec3f d = vec3f(0, 0, distance);
        add_components_vec3f(&pos, &center, &d);

        while (near > distance/100.0f) near /= 10;
        while (far < 2*distance) far *= 2;
        vec3f dir = vec3f(0, 0, -1),
              up = vec3f(0, 1, 0);
        camera_ref cam = make_perspective_cam((char*)"scene-cam", &pos, &dir, &up, 35, float(cmdline.res.x)/cmdline.res.y, near, far);
        use_camera(cam);
        cgl_cam_move_factor = distance/25;
    }

	class default_scene : public scene {
	public:
		default_scene() : scene("default") {
		}
		virtual void load() {
			vec3f min, max;
			the_scene = make_graph_scene("default");
			setup_scene(cmdline.filename, 20, min, max);
		}
	};

	std::map<std::string, scene*> scene::scenes;
	scene* scene::selected = 0;

	class default_scene default_scene;
}


/* vim: set foldmethod=marker: */

