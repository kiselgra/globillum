(let ((home (getenv "HOME")))
  (add-model (string-append home "/render-data/models/sponza.noflag.obj") :type :obj :is-base #t)
  ;(add-model (string-append home "/render-data/models/sponza.noflag.obj") :type :obj :is-base #f)
  (add-model (string-append home "/render-data/models/bunny-70k.obj") :type :obj :trafo '(1 0 0 500
											  0 0 1   0
											  0 1 0   0
											  0 0 0   1))
)
