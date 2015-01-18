(use-modules (ice-9 optargs))
(read-set! keywords 'prefix)

(define (row-maj->col-maj-str trafo)
  (let ((e (lambda (n) (list-ref trafo n))))
    (format #f "~a ~a ~a ~a ~a ~a ~a ~a ~a ~a ~a ~a ~a ~a ~a ~a"
	    (e 0) (e 4) (e 8) (e 12)
	    (e 1) (e 5) (e 9) (e 13)
	    (e 2) (e 6) (e 10) (e 14)
	    (e 3) (e 7) (e 11) (e 15))))

(define (row-maj->row-maj-str trafo)
  (let ((e (lambda (n) (list-ref trafo n))))
    (format #f "~{~a~^ ~}" trafo)))

(define* (add-model name
		    :key type (is-base #f) (trafo '(1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1))
		    (disp "") (proxy "") (spec "") (occl "") (pose ""))
  (cond ((eq? type :obj)     (add-model% name 0 is-base (row-maj->row-maj-str trafo) ""   proxy spec occl pose))
	((eq? type :subd)    (add-model% name 1 is-base (row-maj->row-maj-str trafo) disp proxy spec occl pose))
	((eq? type :subdobj) (add-model% name 2 is-base (row-maj->row-maj-str trafo) disp proxy spec occl pose))
	(else (error "unknown model type: " type))))

