(let ((home (getenv "HOME")))
  (add-model "boxes.obj" :type :obj :is-base #t)
  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/bc_body_trexAA.ptx"
	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/bc_body_trexAA.ptx"
	     :proxy "/tmp/trex.obj")
  (subd-tess 3 5)	; use normal subdiv for 3 levels, quantization for 5 levels.
)
