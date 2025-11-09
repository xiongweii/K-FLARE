### Algorithm 1: Weight-Learning Procedure for Evidence-Balancing Mechanism

**Input**: Training dataset $  D_{\text{train}}  $, Number of folds $  K=5  $

**Output**: Optimized weights $  w^* = (w_{\text{nn}}^*, w_{\text{nli}}^*, w_{\text{llm}}^*)  $



1. Split $  D_{\text{train}}  $ into $  K  $ mutually exclusive folds: $  F_1, F_2, \dots, F_K  $

2. For $  k = 1  $ to $  K  $:


   1. $  D_{\text{val}} \leftarrow F_k  $  // Use $  k  $-th fold as validation set

   2. $  D_{\text{train}_k} \leftarrow D_{\text{train}} \setminus F_k  $  // Remaining folds as training subset

   3. Train the Neural Fusion Module on $  D_{\text{train}_k}  $

   4. Generate outputs for all samples in $  D_{\text{val}}  $:

* Neural network prediction: $  p_{\text{nn}}  $

* NLI evidence score: $  p_{\text{nli}}  $

* LLM confidence score: $  c_{\text{llm}}  $

1. Combine validation outputs from all $  K  $ folds into a unified validation result set

2. Define cross-entropy objective function $  \mathcal{L}_{\text{CE}}(w)  $ based on Eq. B.1, using:

* Combined validation outputs ($  p_{\text{nn}}, p_{\text{nli}}, c_{\text{llm}}  $)

* Corresponding ground-truth labels $  y_{\text{val}}  $

1. Solve for optimal weights by minimizing the objective function:

$ 
   w^* = \arg \min_{w} \mathcal{L}_{\text{CE}}(p_{\text{final}}(w), y_{\text{val}})
    $

where $  p_{\text{final}}(w)  $ denotes the weighted fused prediction
