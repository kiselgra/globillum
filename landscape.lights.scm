(light "area"
       :type :rect
       :color (list  1400 840 520)
       :pos   (list 990 370  -800)
       :dir   (list   -1  -0.5  -0.25)
       :up    (list   0   0  1)
       :dim   (list 150 150))

;(light "area2"
;       :type :rect
;       :color (list  90  70 50)
;       :pos   (list   0 100  0)
;       :dir   (list   0  -1  0)
;       :up    (list   0   0  1)
;       :dim   (list 100 100))

(let ((home (getenv "HOME")))
  (light "sky"
         :type :sky
   ;;      :map "glacier.exr" 
      :map (string-append home "/render-data/images/skylight-sunset.exr")
         :diameter 1000
         :scale 0.5))
