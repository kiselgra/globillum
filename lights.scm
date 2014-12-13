(use-modules (ice-9 optargs))
(read-set! keywords 'prefix)

(define* (light name :key type up color pos dir dim map diameter scale)
  (cond ((eq? type :rect) 
         (add-rectlight name pos dir up color (car dim) (cadr dim)))
        ((eq? type :sky)
         (add-skylight name map diameter scale))
        (else (error "unknown light type: " type))))


