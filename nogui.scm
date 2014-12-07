(define x-res 1)
(define y-res 1)

(use-modules (ice-9 receive))

(use-camera (make-perspective-camera "cam" (list 0 0 0) (list 0 0 -1) (list 0 1 0) 35 (/ x-res y-res) 1 10000))
