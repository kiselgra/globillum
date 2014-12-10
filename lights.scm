(use-modules (ice-9 optargs))
(read-set! keywords 'prefix)

(define* (light name :key type up color pos dir dim)
  (cond ((eq? type :rect) (add-rectlight name pos dir up color (car dim) (cadr dim)))
        (else (error "unknown light type: " type))))

