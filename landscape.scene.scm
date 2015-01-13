(let ((home (getenv "HOME")))
  (add-model (string-append home "/globillum/globillum/landscape.obj") :type :obj :is-base #t)
;  (add-model (string-append "/tmp/rock.obj") : type :obj)
  (add-model (string-append "/tmp/bobble-tree.obj") : type :obj)
  (add-model "/tmp/ram/trex/COLOR/hc_tongue_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hc_tongue_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/al_1claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/al_1claw_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/al_2claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/al_2claw_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/ar_1claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/ar_1claw_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/ar_2claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/ar_2claw_trexAA.ptx")
 (add-model "/tmp/ram/trex/COLOR/hc_dnteeth_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hc_dnteeth_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/hc_gums_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hc_gums_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/hc_upteeth_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hc_upteeth_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/hl_cornea_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hl_cornea_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/hr_cornea_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hr_cornea_trexAA.ptx")
  (add-model "/tmp/ram/trex/COLOR/hl_sclera_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hl_sclera_trexAA.ptx")
 (add-model "/tmp/ram/trex/COLOR/hr_sclera_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/hr_sclera_trexAA.ptx")
 (add-model "/tmp/ram/trex/COLOR/ll_1claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/ll_1claw_trexAA.ptx")
 (add-model "/tmp/ram/trex/COLOR/ll_2claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/ll_2claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/ll_3claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/ll_3claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/ll_4claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/ll_4claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/lr_1claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_1claw_trexAA.ptx")
 (add-model "/tmp/ram/trex/COLOR/lr_2claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_2claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/lr_3claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_3claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/lr_4claw_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/lr_4claw_trexAA.ptx")
(add-model "/tmp/ram/trex/COLOR/bc_body_trexAA.ptx"
	     :type :subd :disp "/tmp/ram/trex/DISP8/bc_body_trexAA.ptx"
	     :proxy "/tmp/trex.obj")
(displacement-scale 3)
	(subd-tess 2 2)
	(dof-config 0.001 900 5)
	(path-samples 64)
	(path-length 5)
	(light-samples 1) ; per direct lighting path
;  (add-model (string-append home "/render-data/models/bunny-70k.obj") :type :obj :trafo '(1 0 0 500
;											  0 0 1   0
;											  0 1 0   0
;											  0 0 0   1))
)
