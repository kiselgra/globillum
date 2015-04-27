(let ((home (getenv "HOME")))
  (add-model (string-append home "/render-data/models/sponza.noflag.obj") :type :obj :is-base #t)
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hc_tongue_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hc_tongue_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/bc_body_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/bc_body_trexAA.ptx"
;	     :proxy "/tmp/trex.obj")
;  (add-model (string-append home "/render-data/models/bunny-70k.obj") :type :obj :trafo '(1 0 0 500
;											  0 0 1   0
;											  0 1 0   0
;											  0 0 0   1))
	(subd-tess 2 2)
	(dof-config 5 900 5)
	(path-samples 64)
	(path-length 3)
	(light-samples 1) ; per direct lighting path
	(integrator "hybrid_pt")
	;(integrator "hybrid_area_lights")
)
