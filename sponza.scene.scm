(let ((home (getenv "HOME")))
  (add-model (string-append home "/render-data/models/sponza.noflag.obj") :type :obj :is-base #t)
  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/bc_body_trexAA.ptx" :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/bc_body_trexAA.ptx")
  )
