(let ((home (getenv "HOME")))
(add-model "plainSphere.obj" :type :obj :is-base #t)
;  (add-model (string-append home "/globillum/globillum/landscape.obj") :type :obj :is-base #t)
;  (add-model (string-append "/tmp/rock.obj") : type :obj)
;  (add-model (string-append "/tmp/bobble-tree.obj") : type :obj)
(add-model "/share/space1/sihyscha/TurtleBarbarian/color_cloth.ptx"
	   :type :subd 
	   :disp "/share/space1/sihyscha/TurtleBarbarian/displacement_cloth.ptx"
	   :spec "/share/space1/sihyscha/TurtleBarbarian/specular_cloth.ptx"
	   :occl "/share/space1/sihyscha/TurtleBarbarian/occlusion_cloth.ptx"
	   :pose "/share/space1/sihyscha/TurtleBarbarian/anim_cloth8/003.obj"
	   :proxy "/share/space1/sihyscha/turtle_barbarian.obj")
(displacement-scale 1)
	(subd-tess 1 2)
	(dof-config 0 60 5)
	(path-samples 2)
	(path-length 4)
;	(render_error_image 1)
	(light-samples 1) ; per direct lighting path
;  (add-model (string-append home "/render-data/models/bunny-70k.obj") :type :obj :trafo '(1 0 0 500
;											  0 0 1   0
;											  0 1 0   0
;											  0 0 0   1))
)
