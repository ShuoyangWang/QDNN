# SCAD-Penalized Partially Linear Quantile Regression with DNN Nonlinear Component

This repository contains a cleaned computational implementation of the SCAD-penalized partially linear quantile-regression framework described by Algorithm 1 and Algorithm 2.

The working model is

```text
y = x_select %*% alpha + x_keep %*% gamma + f_W(z) + error
```

where:

- `x_select` contains linear covariates subject to SCAD selection;
- `x_keep` contains optional linear covariates that are always included and are not selected;
- `z` contains covariates entering the nonlinear DNN component;
- `alpha` is the sparse selected linear coefficient vector;
- `gamma` is the unpenalized kept linear coefficient vector;
- `f_W(z)` is the neural-network nonlinear component.

For compatibility with the original simulation code, use the following mapping:

```r
x_select = data$M   # old mediator block; now selected linear covariates
x_keep   = data$x   # old exposure block; now kept unpenalized covariates
z        = data$z   # nonlinear covariates
y        = data$y
```

The original `datagen.R` functions generate outcomes of the form

```text
y = M %*% alpha0 + x %*% alpha1 + f2_z + error
```

so the above mapping is the intended mapping from the old mediation notation to the current partially linear notation.

## Files

### 1. `estimation_scad_plqr_dnn.R`

Implements the computational version of Algorithm 1:

```text
Adam for SCAD-penalized partially linear quantile regression.
```

Main function:

```r
fit_scad_plqr_dnn()
```

This function fits

```text
y = x_select %*% alpha + x_keep %*% gamma + f_W(z) + error
```

using Keras/Adam, quantile check loss, and SCAD regularization on `alpha` only.

It tunes the SCAD lambda candidate by HBIC over `scad_ids`, then refits the final model at the selected lambda.

Important active-set rule:

```text
If hidden_length == 1:
    active_set = {j: alpha_hat[j] != 0}

If hidden_length > 1:
    active_set = {j: abs(alpha_hat[j]) >= selection_threshold}
```

This matches the intended algorithmic rule: shallow networks use exact sparsity, while deeper networks use a threshold.

### 2. `inference_rank_score_plqr_dnn.R`

Implements the rank score test for crucial linear covariates.

Main functions:

```r
rank_score_test_plqr_dnn()
rank_score_core_plqr()
```

The test supports two cases:

```r
target_from = "select"  # test columns of x_select
target_from = "keep"    # test columns of x_keep
```

The inference file inherits `active_set` from the estimation result. It does not re-select variables using a separate rule. If an old fit object does not contain `active_set`, the inference file reconstructs it using the same depth-dependent rule used in the estimation file.

The rank score statistic is

```text
T_n = S_n^T V_n^{-1} S_n,
```

where `S_n` is formed from the quantile rank scores under the null-restricted model and `V_n` is estimated from the orthogonalized auxiliary regressors.

### 3. `depth_width_selection_scad_plqr_dnn.R`

Implements Algorithm 2:

```text
Depth-and-width selection for SCAD-penalized partially linear quantile regression.
```

Main function:

```r
depth_width_select_scad_plqr()
```

This wrapper:

1. randomly splits the data into training, validation, and testing subsets;
2. obtains preliminary linear SCAD quantile-regression estimates;
3. fits a shallow `L = 1` model with `N_max = max(N_width)`;
4. computes validation discrepancies `D1` and `D2`;
5. chooses either the shallow branch or the deep branch;
6. selects the best architecture by validation quantile loss;
7. refits the final model on training plus validation data;
8. returns the final coefficients, active set, network weights, selected depth, selected width, validation results, and test loss.

The preliminary linear estimator is implemented through Keras/Adam for consistency with Algorithm 1. A statistician may replace the helper `fit_scad_linear_qr()` with a `quantreg::rq`-based quantile-regression routine, for example inside a local-linear-approximation or coordinate-descent implementation for SCAD. Note that `quantreg::rq()` itself is unpenalized unless additional penalty machinery is supplied.

## Required auxiliary files

The code assumes the following auxiliary files are available or already sourced/generated:

```text
SCAD.R       # generates scad1.R, ..., scad8.R
scad1.R      # generated SCAD penalty file
...
scad8.R      # generated SCAD penalty file
datagen.R    # optional simulation data generator
```

The original `SCAD.R` generates SCAD penalty functions with lambda candidates

```r
c(0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
```

and SCAD parameter `a = 3.7`.

## Required R packages

```r
keras
tensorflow
reticulate
mvtnorm    # only needed for the supplied datagen.R
MASS       # only needed as fallback if V_n is singular in inference
```

The code follows the original Keras/TensorFlow style. Because R Keras versions can be sensitive, the implementation intentionally avoids unnecessary modernization.

## Basic workflow

```r
library(keras)
library(tensorflow)
library(reticulate)

use_python(paste(Sys.getenv("CONDA_PREFIX"), "bin/python", sep = "/"))

source("SCAD.R")
source("datagen.R")
source("estimation_scad_plqr_dnn.R")
source("inference_rank_score_plqr_dnn.R")
source("depth_width_selection_scad_plqr_dnn.R")
```

### Example using the original `datagen_r8_test_direct()`

```r
set.seed(1)

data <- datagen_r8_test_direct(
  n = 100,
  p = 200,
  q = 1,
  r = 8,
  type = 1,
  exppar = 2
)
```

### Step 1: architecture selection and final estimation

```r
result <- depth_width_select_scad_plqr(
  x_select = data$M,
  x_keep = data$x,
  z = data$z,
  y = data$y,
  tau = 0.5,

  L_deep = c(2, 3),
  N_width = c(50, 100),
  kappa = 1,
  selection_threshold = 1e-2,

  scad_dir = "~/QMDNN/auxiliary",
  scad_ids = 1:8,

  train_prop = 0.5,
  val_prop = 0.25,
  test_prop = 0.25,

  epochs = 1000,
  batch_size = 4,
  dropout_shallow = 0.5,
  dropout_deep = 0.7,
  seed = 123,
  verbose = 1
)
```

Useful outputs:

```r
result$alpha_hat              # selected linear coefficients
result$gamma_hat              # always-kept linear coefficients
result$active_set             # selected active covariates
result$selected_architecture  # selected L and N
result$architecture_results   # validation loss table
result$test_loss              # test quantile loss
```

### Step 2: rank score inference

To test a kept covariate, which is the closest analog of the old direct-effect test:

```r
test_keep <- rank_score_test_plqr_dnn(
  y = data$y,
  x_select = data$M,
  x_keep = data$x,
  z = data$z,
  selected_fit = result$final_fit,
  target_from = "keep",
  target_index = 1,
  tau = 0.5,
  epochs = 1000,
  batch_size = 4,
  verbose = 1
)

test_keep$stat
test_keep$p_value
```

To test selected/scannable covariates:

```r
test_select <- rank_score_test_plqr_dnn(
  y = data$y,
  x_select = data$M,
  x_keep = data$x,
  z = data$z,
  selected_fit = result$final_fit,
  target_from = "select",
  target_index = c(1, 2, 3),
  tau = 0.5,
  epochs = 1000,
  batch_size = 4,
  verbose = 1
)

test_select$stat
test_select$p_value
```

## Alignment with the algorithms

### Algorithm 1

The estimation file implements:

```text
minimize quantile check loss + SCAD penalty on alpha
```

using Keras automatic differentiation and Adam. The Adam moment updates are handled internally by `optimizer_adam()`.

The implemented model is

```text
y = x_select %*% alpha + x_keep %*% gamma + f_W(z) + error.
```

If the paper algorithm omits `x_keep` for notation simplicity, set `x_keep = NULL`.

### Algorithm 2

The depth-width selection file implements:

```text
training / validation / testing split
linear preliminary fit
shallow L = 1 diagnostic fit
D1 and D2 discrepancy calculation
branch selection using D1 >= kappa * D2
architecture selection by validation quantile loss
final refit on training + validation data
test loss on testing data
```

The returned final active set follows the algorithm:

```text
If L_star == 1:
    A_hat = {j: alpha_hat[j] != 0}

If L_star > 1:
    A_hat = {j: abs(alpha_hat[j]) >= selection_threshold}
```

## Important notes

1. `SCAD.R` must be run before fitting so that `scad1.R`, ..., `scad8.R` exist in `scad_dir`.
2. The code uses `kernel_constraint = constraint_maxnorm(...)` for the hidden layers. If your older Keras environment only works with the original syntax, replace this argument with the syntax used in your old working code.
3. The old variables `M`, `x`, and `z` are no longer used as model notation in the new files. They only appear in examples showing how to map the old data generator into the new framework.
4. The inference file uses residuals as `epsilon_hat = y - fitted`, matching the rank score formula.
5. The projection residuals are computed as `D = x_interest - fitted projection`, matching the definition of the auxiliary regressors.
