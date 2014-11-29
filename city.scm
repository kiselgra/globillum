;;; default config for c based viewer.
;;;

(format #t "Entering ~a.~%" (current-filename))

;;; modules
(use-modules (ice-9 receive))

;;; actual code

(define x-res 1)
(define y-res 1)

(receive (x y w h) (get-viewport)
  (format #t "~a x ~a~%" w h)
  (set! x-res w)
  (set! y-res h))

(let ((home (getenv "HOME")))
  (append-image-path (string-append home "/render-data/images"))
  (append-image-path (string-append home "/render-data/images/wikimedia"))
  (append-image-path (string-append home "/render-data/images/sponza")))

(defmacro cmdline (x)
  `(query-cmdline ',x))

(define use-graph #t)

(define the-scene (if use-graph (make-graph-scene "default")
                                (make-scene "default")))

(define (make-de name mesh material bbmin bbmax)
  ;(material-use-stock-shader! material)
  (let* ((shader -1);(material-shader material))
		 (de (make-drawelement name mesh shader material)))
	(prepend-uniform-handler de 'default-matrix-uniform-handler)
	(prepend-uniform-handler de 'default-material-uniform-handler)
    (add-drawelement-to-scene the-scene de)
    (drawelement-bounding-box! de bbmin bbmax)
    de))

(define (make-de-idx name mesh material pos len bbmin bbmax)
  (let ((de (make-de name mesh material bbmin bbmax)))
    (drawelement-index-buffer-range! de pos len)
    de))

(let ((fallback (make-material "fallback" (list 1 0 0 1) (list 1 0 0 1) (list 0 0 0 1))))
  (receive (min max) (if (not (string=? (basename (cmdline model) "obj") (basename (cmdline model))))
                         (if use-graph
                             (load-objfile-and-create-objects-with-single-vbo (cmdline model) (cmdline model) make-de-idx fallback (cmdline merge-factor))
                             (load-objfile-and-create-objects-with-separate-vbos (cmdline model) (cmdline model) make-de fallback))
                         (load-model-and-create-objects-with-separate-vbos (cmdline model) (cmdline model) make-de fallback))
    (let* ((near 1)
		   (far 1000)
		   (diam (vec-sub max min))
		   (diam/2 (vec-div-by-scalar diam 2))
		   (center (vec-add min diam/2))
		   (distance (vec-length diam))
		   (pos (vec-add center (make-vec 0 0 distance))))
	  (while (> near (/ distance 100))
	    (set! near (/ near 10)))
	  (while (< far (* distance 2))
	    (set! far (* far 2)))
	  (let ((cam (make-perspective-camera "cam" pos (list 0 0 -1) (list 0 1 0) 35 (/ x-res y-res) near far)))
        (use-camera cam))
      (set-move-factor! (/ distance 20)))))

(gl:enable gl#depth-test)


(format #t "scene specific setup~%")
(define home (getenv "HOME"))

(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/LGDV_Logo_Vertical_final_proxy_Circle.003_Neon_internal"))) (list 10 0 0 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/LGDV_Logo_Horizontal_proxy_Circle.002_Neon"))) (list 10 0 0 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/Kino_text_proxy_Mesh_Kino_neon"))) (list 1 8 1 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/Kino_text_proxy_Mesh_Kino_neon_green"))) (list 0 10 0 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/Kino_text_proxy_Mesh_Kino_neon_orange"))) (list 5 4 0 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/Kino_text_proxy_Mesh_Kino_neon_blue"))) (list 1 1 8 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/lantern_proxy.007_Circle.009_in_light"))) (list 5 5 3 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/Gyrocopter_proxy_Circle.003_mat_gyrocopter_lights.internal"))) (list 2.4 5 8.8 1))
(set-material-diffuse-color! (drawelement-material (find-drawelement (string-append home "/lgdv-scene/city.lowrestex.obj/Skyscraper_final_proxy_Plane.004_Light_windows_internal"))) (list 3 3 6 1))



(define (setup-lights)
  (let ((deferred (find-scene "default"))
	(forward (find-scene "forward")))
    (let ((hemi (make-hemispherical-light "hemi" (find-framebuffer "gbuffer") (list 0 1 0)))
          	(lamp_spot_0 (make-spotlight "lamp_spot_0" (find-framebuffer "gbuffer") (list -157.237840 439.278126 118.351066) (list 0.000000 -1.000000 -0.000000) (list 0 1 0) 120))
		(lamp_spot_1 (make-spotlight "lamp_spot_1" (find-framebuffer "gbuffer") (list 2303.620338 1533.224106 -147.046709) (list 0.524193 -0.849082 0.065438) (list 0 1 0) 58))
		(lamp_spot_2 (make-spotlight "lamp_spot_2" (find-framebuffer "gbuffer") (list 181.114006 439.278126 76.996291) (list 0.159791 -0.984808 0.067974) (list 0 1 0) 120))
		(lamp_spot_3 (make-spotlight "lamp_spot_3" (find-framebuffer "gbuffer") (list -23.876405 439.278126 -195.347464) (list -0.021028 -0.984808 -0.172370) (list 0 1 0) 120))
		(lamp_spot_4 (make-spotlight "lamp_spot_4" (find-framebuffer "gbuffer") (list 1476.123619 439.278126 -195.347464) (list -0.021028 -0.984808 -0.172370) (list 0 1 0) 120))
		(lamp_spot_5 (make-spotlight "lamp_spot_5" (find-framebuffer "gbuffer") (list 1681.114006 439.278126 76.996291) (list 0.159791 -0.984808 0.067974) (list 0 1 0) 120))
		(lamp_spot_6 (make-spotlight "lamp_spot_6" (find-framebuffer "gbuffer") (list 1342.762184 439.278126 118.351066) (list -0.138763 -0.984808 0.104396) (list 0 1 0) 120))
		(lamp_spot_7 (make-spotlight "lamp_spot_7" (find-framebuffer "gbuffer") (list 2842.762184 439.278126 118.351066) (list -0.138763 -0.984808 0.104396) (list 0 1 0) 120))
		(lamp_spot_8 (make-spotlight "lamp_spot_8" (find-framebuffer "gbuffer") (list 3181.114006 439.278126 76.996291) (list 0.159791 -0.984808 0.067974) (list 0 1 0) 120))
		(lamp_spot_9 (make-spotlight "lamp_spot_9" (find-framebuffer "gbuffer") (list 2976.123619 439.278126 -195.347464) (list -0.021028 -0.984808 -0.172370) (list 0 1 0) 120))
		(lamp_spot_10 (make-spotlight "lamp_spot_10" (find-framebuffer "gbuffer") (list 2573.617554 870.504189 -102.865469) (list -0.723843 -0.673595 -0.149401) (list 0 1 0) 58))
		(lamp_spot_11 (make-spotlight "lamp_spot_11" (find-framebuffer "gbuffer") (list 4456.742096 503.332567 22.523242) (list 0.000000 -1.000000 -0.000000) (list 0 1 0) 120))
		(lamp_spot_12 (make-spotlight "lamp_spot_12" (find-framebuffer "gbuffer") (list 5939.310837 479.485941 11.121880) (list 0.000000 -1.000000 -0.000000) (list 0 1 0) 120)))
		(set-light-color! lamp_spot_0 (list 0.786470 0.758979 0.407962))
		;(set-light-color! lamp_spot_1 (list 1.000000 1.000000 1.000000))
		(set-light-color! lamp_spot_2 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_3 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_4 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_5 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_6 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_7 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_8 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_9 (list 0.786470 0.758979 0.407962))
		;(set-light-color! lamp_spot_10 (list 1.119222 1.550980 2.000000))
		(set-light-color! lamp_spot_11 (list 0.786470 0.758979 0.407962))
		(set-light-color! lamp_spot_12 (list 0.786470 0.758979 0.407962))
		(add-light-to-scene deferred lamp_spot_0)
		;(add-light-to-scene deferred lamp_spot_1)
		(add-light-to-scene deferred lamp_spot_2)
		(add-light-to-scene deferred lamp_spot_3)
		(add-light-to-scene deferred lamp_spot_4)
		(add-light-to-scene deferred lamp_spot_5)
		(add-light-to-scene deferred lamp_spot_6)
		(add-light-to-scene deferred lamp_spot_7)
		(add-light-to-scene deferred lamp_spot_8)
		(add-light-to-scene deferred lamp_spot_9)
		;(add-light-to-scene deferred lamp_spot_10)
		(add-light-to-scene deferred lamp_spot_11)
		(add-light-to-scene deferred lamp_spot_12)
		(add-light-to-scene forward lamp_spot_0)
		;(add-light-to-scene forward lamp_spot_1)
		(add-light-to-scene forward lamp_spot_2)
		(add-light-to-scene forward lamp_spot_3)
		(add-light-to-scene forward lamp_spot_4)
		(add-light-to-scene forward lamp_spot_5)
		(add-light-to-scene forward lamp_spot_6)
		(add-light-to-scene forward lamp_spot_7)
		(add-light-to-scene forward lamp_spot_8)
																																																																																																																				(add-light-to-scene forward lamp_spot_9)
		;(add-light-to-scene forward lamp_spot_10)
		(add-light-to-scene forward lamp_spot_11)
		(add-light-to-scene forward lamp_spot_12)

		(set-light-color! hemi (list 1 1 1.5))
	      	(add-light-to-scene deferred hemi)
	      	(add-light-to-scene forward hemi)
)))


(define heli '())
(define lamp #f)

(let loop ((des (list-drawelements))
           (anim-des '()))
  (if (null? des)
      (set! heli anim-des)
      (if (or (string-contains-ci (car des) "heli")
              (string-contains-ci (car des) "gyro"))
          (loop (cdr des) (cons (find-drawelement (car des)) anim-des))
          (if (string-contains-ci (car des) "lantern_proxy")
              (begin (set! lamp (find-drawelement (car des)))
                     (loop (cdr des) anim-des))
              (loop (cdr des) anim-des)))))

(define (move-lamp x y z)
  (let ((trafo (de-trafo lamp)))
    (mset! trafo 3 0 (+ (mref trafo 3 0) x))
    (mset! trafo 3 1 (+ (mref trafo 3 1) y))
    (mset! trafo 3 2 (+ (mref trafo 3 2) z))
    (set-de-trafo! lamp trafo)))

          
(defmacro dotimes (v n . what)
  `(let dotimes-loop ((,v 0))
     (if (not (= ,v ,n))
         (begin
           ,@what
           (dotimes-loop (1+ ,v))))))

(define (stdkbx str)
  (stdkb (string-ref str 0)))

; at g5,
; move lamp post
; delta=60
; ap=1
(let ((ca (make-command-animation "focus"))
      (s1 23)
      (s2 160)
      (o 0))
  (add-node-to-command-animation ca "(set-move-factor! .25)" 0)
  (add-node-to-command-animation ca "(fod 70)" 0)
  (dotimes i 30 (add-node-to-command-animation ca "(stdkbx \"a\")" (/ i 5)))
  (dotimes i 60 (add-node-to-command-animation ca "(stdkbx \"d\")" (+ (/ i 5) 6)))
  (dotimes i 30 (add-node-to-command-animation ca "(stdkbx \"a\")" (+ (/ i 5) 18)))
  (set! o 24)
  (let loop ((step 0) (steps s1))
    (add-node-to-command-animation ca (format #f "(let ((f ~a)) (fod f)(display 'y)(display f)(newline))" (+ 70 (* 10 step))) (+ o step))
    (if (< step steps) (loop (1+ step) steps)))
  (set! o (+ o s1))
  (let loop ((step 0) (steps s2))
    (add-node-to-command-animation ca (format #f "(let ((f ~a)) (fod f)(display 'z)(display f)(newline))" (+ 300 (* 2 step))) (+ o step))
    (if (< step steps) (loop (1+ step) steps)))
  (set! o (+ o s2))
  (dotimes i 30 (add-node-to-command-animation ca "(stdkbx \"a\")" (+ o (/ i 5))))
  (dotimes i 60 (add-node-to-command-animation ca "(stdkbx \"d\")" (+ o (/ i 5) 6)))
  (dotimes i 30 (add-node-to-command-animation ca "(stdkbx \"a\")" (+ o (/ i 5) 18)))
  (change-command-animation-speed! ca 6)
  )



(format #t "Leaving ~a.~%" (current-filename))
