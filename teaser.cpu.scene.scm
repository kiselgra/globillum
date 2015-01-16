(let ((home (getenv "HOME")))
  (add-model "plainSphere.obj" :type :obj :is-base #t)
  (add-model "/tmp/trex.obj" :type :obj)
					;  (add-model (string-append home "/globillum/globillum/landscape.obj") :type :obj :is-base #t)
					;  (add-model (string-append "/tmp/rock.obj") : type :obj)
					;  (add-model (string-append "/tmp/bobble-tree.obj") : type :obj)
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hc_tongue_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hc_tongue_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/al_1claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/al_1claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/al_2claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/al_2claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/ar_1claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/ar_1claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/ar_2claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/ar_2claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hc_dnteeth_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hc_dnteeth_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hc_gums_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hc_gums_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hc_upteeth_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hc_upteeth_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hl_cornea_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hl_cornea_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hr_cornea_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hr_cornea_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hl_sclera_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hl_sclera_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/hr_sclera_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/hr_sclera_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/ll_1claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/ll_1claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/ll_2claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/ll_2claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/ll_3claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/ll_3claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/ll_4claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/ll_4claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/lr_1claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/lr_1claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/lr_2claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/lr_2claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/lr_3claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/lr_3claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/lr_4claw_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/lr_4claw_trexAA.ptx")
;  (add-model "/share/space1/simaprus/TREX/TREX_ptx/COLOR/bc_body_trexAA.ptx"
;	     :type :subd :disp "/share/space1/simaprus/TREX/TREX_ptx/DISP8/bc_body_trexAA.ptx"
;	     :proxy "/tmp/trex.obj")

  (displacement-scale 3)
  (subd-tess 2 2)
  (dof-config 0 60 5)
  (path-samples 32)
  (path-length 5)
  (integrator "cpu_pt")
  (light-samples 1) ; per direct lighting path

  (light "area"
	 :type :rect
	 :color (list  1400 840 520)
	 :pos   (list 990 370  -800)
	 :dir   (list   -1  -0.5  -0.25)
	 :up    (list   0   0  1)
	 :dim   (list 150 150))

  (light "sky"
         :type :sky
;        :map "pisa.exr" 
	 :map (string-append home "/render-data/images/skylight-sunset.exr")
         :diameter 1000
         :scale 1)
  )
