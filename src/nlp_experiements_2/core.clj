(ns nlp-experiements-2.core
  (:require [clojure.string :as s]))

; Pre-process text

(def text "Kantian ethics refers to a deontological ethical theory developed by German philosopher Immanuel Kant that is based on the notion that: \"It is impossible to think of anything at all in the world, or indeed even beyond it, that could be considered good without limitation except a good will.\" The theory was developed as a result of Enlightenment rationalism, stating that an action can only be right if its maxim—the principle behind it—is duty to the moral law, and arises from a sense of duty in the actor.\n\nCentral to Kant's construction of the moral law is the categorical imperative, which acts on all people, regardless of their interests or desires. Kant formulated the categorical imperative in various ways. His principle of universalizability requires that, for an action to be permissible, it must be possible to apply it to all people without a contradiction occurring. Kant's formulation of humanity, the second section of the categorical imperative, states that as an end in itself, humans are required never to treat others merely as a means to an end, but always as ends in themselves. The formulation of autonomy concludes that rational agents are bound to the moral law by their own will, while Kant's concept of the Kingdom of Ends requires that people act as if the principles of their actions establish a law for a hypothetical kingdom. Kant also distinguished between perfect and imperfect duties. Kant used the example of lying as an application of his ethics: because there is a perfect duty to tell the truth, we must never lie, even if it seems that lying would bring about better consequences than telling the truth. Likewise, a perfect duty (e.g. the duty not to lie) always holds true; an imperfect duty (e.g., the duty to give to charity) can be made flexible and applied in particular time and place.\n\nThose influenced by Kantian ethics include social philosopher Jürgen Habermas, political philosopher John Rawls, and psychoanalyst Jacques Lacan. German philosopher G. W. F. Hegel criticised Kant for not providing specific enough detail in his moral theory to affect decision-making and for denying human nature. The Catholic Church has criticized Kant's ethics as contradictory, and regards Christian ethics as more compatible with virtue ethics. German philosopher Arthur Schopenhauer, arguing that ethics should attempt to describe how people behave, criticised Kant for being prescriptive. Marcia Baron has defended the theory by arguing that duty does not diminish other motivations.\n\nThe claim that all humans are due dignity and respect as autonomous agents necessitates that medical professionals should be happy for their treatments to be performed on anyone, and that patients must never be treated merely as useful for society. Kant's approach to sexual ethics emerged from his view that humans should never be used merely as a means to an end, leading him to regard sexual activity as degrading, and to condemn certain specific sexual practices—for example, extramarital sex. Accordingly, some feminist philosophers have used Kantian ethics to condemn practices such as prostitution and pornography, which treat women as means alone. Kant also believed that, because animals do not possess rationality, we cannot have duties to them except indirect duties not to develop immoral dispositions through cruelty towards them.\n\n")
(defn tokenize [text]
  (re-seq #"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*" text))
(def tokens (pmap s/lower-case
                 (tokenize text)))
(defn generate-maps [tokens]
  (let [lis (map-indexed
              (fn [a b]
                [a (.toLowerCase b)])
              (set tokens))]
    [(into (sorted-map) lis)
     (into (sorted-map)
           (map (fn [[a b]] [(keyword b) a]) lis))]))
(def maps (generate-maps tokens))
(def word_to_id (last maps))
(def id_to_word (first maps))
; Generate training data
(defn generate-contexts [n-tokens tokens idx window]
  (let [range-min (max 0 (- idx window))
        range-max (+ 1 (min
                         (- n-tokens 1)
                         (+ idx window)))]
    (pmap (fn [current]
           (list (nth tokens idx)
                 (nth tokens current)))
         (remove #(== idx %)
                 (range range-min range-max)))))
(defn pairs [tokens n-tokens window]
  (apply concat
         (map
           #(generate-contexts n-tokens tokens % window)
           (range n-tokens))))

(defn one-hot-encode [id, corpus-size]
  (assoc (vec (repeat corpus-size 0)) id 1))

(defn generate-training-data [tokens, word_to_id, window]
  (let [n-tokens (count word_to_id)
        pairs (pairs tokens n-tokens window)]
    (list
      (pmap (fn [[w1 w2]]
             (one-hot-encode
               (word_to_id (keyword w1))
               n-tokens))
           pairs)
      (pmap (fn [[w1 w2]]
             (one-hot-encode
               (word_to_id (keyword w2))
               n-tokens))
           pairs)
      )))
(nth (one-hot-encode (word_to_id :and) 200) (word_to_id :and))

(def data (generate-training-data tokens word_to_id 2))

(let [data data
      X (first data)
      y (second data)]
  ;(dim X)
  )

(defn init-weights [vocab-size n-embedding]
  (hash-map :w1 (repeatedly vocab-size
                            (fn [] (repeatedly n-embedding
                                               #(rand-int 2))))
            :w2 (repeatedly n-embedding
                            (fn [] (repeatedly vocab-size
                                               #(rand-int 2))))))
;(init-weights (count word_to_id), 10)

;;
;; Matrix operations
;;
(defn dot [x y]                                             ; vector dot
  (reduce + (pmap * x y)))

(defn transpose [x]                                         ; transpose mat
  (vec (apply pmap vector x)))

(defn madd [X Y]
  (partition
    (count (first X))
    (map #(+ %1 %2) (flatten X) (flatten Y)))
  )
;(madd [[3 3] [3 4]] [[33 3] [2 4]])

(defn mult [X Y]                                            ; matrix multiplication
  {:pre [(= (count (nth X 0)) (count Y))]}
  (pmap vec
       (partition
         (count (transpose Y))
         (for
           [x X
            y (transpose Y)]
           (dot x y)))))

; (mult `((2 2 3 2) (2 2 3 2)) `((1 1 1 1) (1 1 1 1) (1 1 1 1) (1 1 1 1)))

;
;HELPER: matrix map. maps operation f to each element in X
;
(defn mmap [f X]
  (partition
    (count (first X))
    (pmap f (flatten X))))

(defn softmax [X]
  (let [exp (map (fn [x] (Math/pow x (Math/exp 1)))
                 (flatten X))
        exp_sum (reduce + exp)]
    (partition
      (count (first X))
      (pmap #(/ % exp_sum)
           exp))))
;(softmax [[1 1 2] [1 1 2]])
(defn dim [X]
  [:row (count X) :col (count (first X))])
;(dim [[2 3]])

;
; Forward
;
(defn forward [weights X return_cache]
  (let [a1 (mult X (weights :w1))
        a2 (mult a1 (weights :w2))
        z (softmax a2)]
    (if return_cache
      (hash-map :a1 a1
                :a2 a2
                :z z)
      z)
    )
  (printf "Forwarding..."))
;(count (first (first data)))
;(count word_to_id)
(dim (first data))
(map dim (vals (init-weights (count word_to_id)
                             10)))
(map dim (vals
           (forward (init-weights (count word_to_id)
                                  10)
                    (first data)
                    true)))

;
; loss (cross-entropy)
;
;def cross_entropy(z, y):
;return - np.sum(np.log(z) * y)
(defn loss [z y]
  (reduce + (filter #(not (or (Double/isNaN %)
                              (Double/isInfinite %)))
                    (flatten
                      (map *
                           (flatten (mmap #(Math/log %) z))
                           (flatten y)))))
  (printf "Calculated loss..."))
;(loss [[0.2 0.9 0.1 0] [0.2 0.9 0.1 0]] [[0 1 0 0] [1 0 0 0]])
;(mult [[3 2 1] [3 2 1]] [[1] [3] [2]])

;
; Backward
;
;(def fwd-cached (forward (init-weights (count word_to_id) 10) (first data) true))
(defn backward [weights X, y, alpha]
  (let [
        cache (forward weights X true)
        ;cache fwd-cached
        w1 (:w1 weights)
        w2 (:w2 weights)
        da2 (:z cache)
        dw2 (mult (transpose (:a1 cache)) da2)
        da1 (mult da2 (transpose w2))
        dw1 (mult (transpose X) da1)
        ]
    (assert (dim dw2) (dim w2))                             ; Check if dim calc is correct
    (assert (dim dw1) (dim w1))                             ; Check if dim calc is correct
    [dw1 dw2]
    (hash-map :updated-weights
              (hash-map :w1
                        (mmap #(* (- alpha) %) dw1)
                        :w2
                        (mmap #(* (- alpha) %) dw2))
              :loss
              (loss da2 y))
    (printf "Backpropagated...")))

;(backward (init-weights (count word_to_id) 1) (first data) (second data) 0.05)
(def params [:n 50
             :learning-rate 0.05])
(defn train [weights X y params]
  (loop [model weights
         i 0]
    (if (< i 50)
      (let [b (backward weights X y (:learning-rate params))
            new-weights (:updated-weights b)
            loss (:loss b)]
        (printf "Training at epoch %d, loss: %f" i loss)
        (recur new-weights (inc i)))
      model)))

(train (init-weights (count word_to_id) 1)
       (first data)
       (second data)
       params)