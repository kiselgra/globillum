(if gui
    (load-shader-file "local.shader"))

(use-modules (srfi srfi-1))

(define bookmarks '())

(define (bookmark model name data)
  (set! bookmarks (cons (list model name data)
                        bookmarks)))

(define (bookmarks-for-model model)
  (remove (lambda (x) (not (string=? (first x) model)))
          bookmarks))

(define (list-bookmarks model)
  (map (lambda (x) (second x))
       (bookmarks-for-model model)))

(define (select-bookmark model mark)
  (let* ((marks (bookmarks-for-model model))
         (matching (remove (lambda (x) (not (string-contains (second x) mark)))
                           marks)))
    (if (not (null? matching))
        (let ((match (first matching)))
          (eval `(begin
                   (change-lookat-of-cam! (current-camera) ,@(third match))
                   (recompute-gl-matrices-of-cam (current-camera)))
                (interaction-environment))
          (second match))
        #f)))
    
(if (file-exists? "bookmarks")
    (primitive-load "bookmarks"))


