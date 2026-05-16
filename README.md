# SCAD-Penalized Partially Linear Quantile Regression with Neural Network Nonlinear Component with Deep and Shallow Structure Selection

This repository provides R/Keras code for SCAD-penalized partially linear quantile regression with a neural-network nonlinear component and data-driven shallow/deep structure selection.

The working model is

```text
y = x_select %*% alpha + x_keep %*% gamma + f_W(z) + error
```

where:

- `x_select` contains linear covariates subject to SCAD variable selection;
- `x_keep` contains optional linear covariates that are always included and are not selected;
- `z` contains covariates entering the nonlinear neural-network component;
- `alpha` is the sparse coefficient vector for selected linear covariates;
- `gamma` is the unpenalized coefficient vector for always-kept linear covariates;
- `f_W(z)` is the nonlinear component represented by neural-network weights `W`.

If there are no always-kept linear covariates, set `x_keep = NULL`.

## Files

### `estimation.R`

Main function:

```r
fit_scad_plqr_dnn()
```

This function fits the partially linear quantile-regression model using Keras/Adam, quantile check loss, and a SCAD penalty on `alpha` only.

The function tunes the SCAD penalty candidate by HBIC and then refits the final model at the selected penalty level.

The active-set rule depends on the selected neural-network depth:

```text
If hidden_length == 1:
    active_set = {j: alpha_hat[j] != 0}

If hidden_length > 1:
    active_set = {j: abs(alpha_hat[j]) >= selection_threshold}
```

Returned quantities include:

```r
alpha_hat       # SCAD-penalized selected linear coefficients
gamma_hat       # always-kept unpenalized linear coefficients
active_set      # selected active covariates
y_pred          # fitted values
tuning_best     # selected SCAD penalty file index
HBIC            # HBIC values over candidate penalties
df              # selected degrees of freedom over candidate penalties
model           # returned only when return_model = TRUE
```

### `inference.R`

Main functions:

```r
rank_score_test_plqr_dnn()
rank_score_core_plqr()
```

This file implements a rank-score test for linear covariates in the partially linear quantile-regression model.

The test supports two target types:

```r
target_from = "select"  # test columns of x_select
target_from = "keep"    # test columns of x_keep
```

The inference routine inherits `active_set` from the fitted estimation object. It does not re-select variables with a separate rule. If a fitted object does not contain `active_set`, the routine reconstructs it using the same depth-dependent rule used in `estimation.R`.

The rank-score statistic is

```text
T_n = S_n^T V_n^{-1} S_n,
```

where `S_n` is formed from quantile rank scores under the null-restricted fit and `V_n` is estimated from orthogonalized auxiliary regressors.

### `depth_width_selection.R`

Main function:

```r
depth_width_select_scad_plqr()
```

This wrapper selects the neural-network depth and width, then returns the final refitted model.

The workflow is:

1. split the observations into training, validation, and testing subsets;
2. obtain a preliminary linear SCAD quantile-regression fit;
3. fit a shallow single-hidden-layer neural network at the largest candidate width;
4. compare validation discrepancies between the linear and shallow nonlinear fits;
5. search either the shallow candidate widths or the deeper depth-width grid;
6. select the structure with the smallest validation quantile loss;
7. refit the final model on training plus validation data;
8. compute test quantile loss when a testing subset is retained.

The preliminary linear estimator is implemented through Keras/Adam for consistency with the main nonlinear estimator. Users may replace the helper `fit_scad_linear_qr()` with a `quantreg::rq`-based quantile-regression routine, for example inside a local-linear-approximation or coordinate-descent implementation for SCAD. Note that `quantreg::rq()` itself is unpenalized unless additional penalty machinery is supplied.

Returned quantities include:

```r
alpha_hat              # final selected linear coefficients
gamma_hat              # final always-kept linear coefficients
W_hat                  # final neural-network weights
active_set             # final selected active covariates
L_star                 # selected depth
N_star                 # selected width
selected_architecture  # named vector c(L = L_star, N = N_star)
architecture_results   # validation losses for searched structures
final_fit              # final estimation object
test_loss              # test quantile loss
split                  # training, validation, and testing indices
```

## Auxiliary SCAD penalty files

The estimation routines expect SCAD penalty files in `scad_dir`, with names corresponding to `scad_ids`, such as:

```text
scad1.R
scad2.R
...
```

`SCAD.R` can be used as an example script to generate the corresponding SCAD penalty files. The generated files should define a function named `scad`, which is passed to Keras as the kernel regularizer for the selected linear coefficients.

## Required R packages

```r
keras
tensorflow
reticulate
MASS      # optional fallback for generalized inverse in inference
quantreg  # optional, only if users replace fit_scad_linear_qr()
```

The code intentionally follows a conservative R/Keras style to remain close to working Keras/TensorFlow implementations.

## Basic usage

```r
library(keras)
library(tensorflow)
library(reticulate)

use_python(paste(Sys.getenv("CONDA_PREFIX"), "bin/python", sep = "/"))

source("estimation.R")
source("inference.R")
source("depth_width_selection.R")
```

Run `SCAD.R` or otherwise prepare the SCAD penalty files before fitting. Then define your data objects:

```r
# Required objects:
#   y        : numeric outcome vector of length n
#   x_select : n by p_select matrix of SCAD-selected linear covariates
#   x_keep   : optional n by p_keep matrix of always-kept covariates, or NULL
#   z        : n by r matrix of nonlinear covariates
```

### Estimation with a fixed structure

```r
fit <- fit_scad_plqr_dnn(
  x_select = x_select,
  x_keep = x_keep,
  z = z,
  y = y,
  tau = 0.5,
  scad_dir = "path/to/scad/files",
  scad_ids = 1:8,
  hidden_length = 2,
  width = 100,
  dropout = 0.5,
  epochs = 1000,
  batch_size = 4,
  selection_threshold = 1e-2,
  return_model = TRUE
)

fit$alpha_hat
fit$gamma_hat
fit$active_set
```

### Depth-and-width selection

```r
selected <- depth_width_select_scad_plqr(
  x_select = x_select,
  x_keep = x_keep,
  z = z,
  y = y,
  tau = 0.5,
  L_deep = c(2, 3),
  N_width = c(50, 100),
  kappa = 1,
  selection_threshold = 1e-2,
  scad_dir = "path/to/scad/files",
  scad_ids = 1:8,
  train_prop = 0.5,
  val_prop = 0.25,
  test_prop = 0.25,
  epochs = 1000,
  batch_size = 4,
  dropout_shallow = 0.5,
  dropout_deep = 0.5,
  seed = 123,
  return_all_fits = FALSE
)

selected$selected_architecture
selected$active_set
selected$test_loss
```

### Rank-score inference

Test selected/scannable covariates:

```r
test_select <- rank_score_test_plqr_dnn(
  y = y,
  x_select = x_select,
  x_keep = x_keep,
  z = z,
  selected_fit = selected$final_fit,
  target_from = "select",
  target_index = c(1, 2, 3),
  tau = 0.5,
  hidden_length = selected$L_star,
  width = selected$N_star,
  dropout = 0.5,
  epochs = 1000,
  batch_size = 4
)

test_select$stat
test_select$p_value
```

Test always-kept covariates:

```r
test_keep <- rank_score_test_plqr_dnn(
  y = y,
  x_select = x_select,
  x_keep = x_keep,
  z = z,
  selected_fit = selected$final_fit,
  target_from = "keep",
  target_index = 1,
  tau = 0.5,
  hidden_length = selected$L_star,
  width = selected$N_star,
  dropout = 0.5,
  epochs = 1000,
  batch_size = 4
)

test_keep$stat
test_keep$p_value
```

## Notes

- `x_select`, `x_keep`, `z`, and `y` must have compatible sample sizes.
- The SCAD penalty is applied only to `x_select`.
- The final active set is inherited by the inference routine from the estimation or structure-selection result.
- For a single-hidden-layer fit, active variables are defined by exact nonzero coefficients.
- For deeper fits, active variables are defined by the threshold rule using `selection_threshold`.
- The hidden layers use a max-norm kernel constraint. If your Keras version requires a different argument name, use the syntax supported by your installed Keras/TensorFlow version.
