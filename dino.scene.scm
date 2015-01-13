(let ((home (getenv "HOME")))
  (add-model "/tmp/boxes-d3.obj" :type :obj :is-base #t)
(add-model "/tmp/ram/trex/COLOR/hc_upteeth_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hc_upteeth_trexAA.ptx")
;  (add-model "/tmp/trex.obj" :type :obj)
  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/bc_body_trexAA.ptx"
	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/bc_body_trexAA.ptx"
	     :proxy "/tmp/trex.obj")
   (subd-tess 3 4) ;use normal subdiv for 3 levels, quantization for 5 levels.
	(displacement-scale 3)
)
