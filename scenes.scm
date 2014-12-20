(use-modules (ice-9 optargs))
(read-set! keywords 'prefix)

(define* (add-model name :key type (is-base #f) disp)
  (cond ((eq? type :obj) (add-model% name 0 is-base ""))
	((eq? type :subd) (add-model% name 1 is-base disp))
	(else (error "unknown model type: " type))))

