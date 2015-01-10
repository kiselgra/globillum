;(light "area"
;       :type :rect
;       :color (list  50  50 50)
;       :pos   (list -200 670  0)
;       :dir   (list   0  -1  0)
;       :up    (list   0   0  1)
;       :dim   (list 100 100))

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
         ;:map "cgskies-0319-free.png"
         :map (string-append home "/render-data/images/skylight-morn.exr")
         :diameter 1000
         :scale 10))
