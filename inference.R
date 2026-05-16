# ============================================================
# inference_rank_score_plqr_dnn.R
#
# Rank score test for crucial linear covariates in the current
# partially linear quantile-regression framework.
#
# Model:
#   y = x_select %*% alpha + x_keep %*% gamma + f_W(z) + error
#
# The tested variables are denoted x_interest. Nuisance linear
# variables are denoted x_nuisance. The nonlinear nuisance part is z.
#
# The inference file inherits the active set from the estimation file.
# It does not apply a different selection rule unless selected_fit
# lacks active_set, in which case it reconstructs the active set using
# the same depth-dependent rule.
# ============================================================


# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

.rank_as_matrix <- function(x, name) {
  if (is.null(x)) return(NULL)
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  if (!is.numeric(x)) stop(name, " must be numeric.")
  x
}

.rank_prepare_widths <- function(width, hidden_length) {
  if (length(width) == 1) {
    widths <- rep(width, hidden_length)
  } else {
    widths <- width
  }

  if (length(widths) != hidden_length) {
    stop("If width is a vector, length(width) must equal hidden_length.")
  }

  as.integer(widths)
}

.rank_cbind_nonnull <- function(...) {
  mats <- list(...)
  mats <- mats[!vapply(mats, is.null, logical(1))]
  if (length(mats) == 0) return(NULL)
  do.call(cbind, mats)
}

# ------------------------------------------------------------
# Fit an unpenalized partially linear DNN model.
# Used for:
#   1. null-restricted quantile fit;
#   2. least-squares projection of target covariates.
# ------------------------------------------------------------
fit_unpenalized_pl_dnn <- function(
    x_linear = NULL,
    z,
    y,
    loss_type = c("quantile", "mse"),
    tau = 0.5,
    hidden_length = 2,
    width = 100,
    dropout = 0.5,
    epochs = 1000,
    batch_size = 4,
    verbose = 1
) {

  library(keras)
  library(tensorflow)

  loss_type <- match.arg(loss_type)

  z <- .rank_as_matrix(z, "z")
  x_linear <- .rank_as_matrix(x_linear, "x_linear")
  y <- as.numeric(y)

  n <- length(y)
  r <- ncol(z)
  p_linear <- ifelse(is.null(x_linear), 0, ncol(x_linear))

  if (nrow(z) != n) stop("z and y must have the same number of observations.")
  if (!is.null(x_linear) && nrow(x_linear) != n) {
    stop("x_linear and y must have the same number of observations.")
  }

  widths <- .rank_prepare_widths(width, hidden_length)

  tilted_loss <- function(qtau, y_true, y_pred) {
    e <- y_true - y_pred
    k_mean(k_maximum(qtau * e, (qtau - 1) * e), axis = 2)
  }

  x_train <- vector("list", p_linear + 1)

  if (p_linear > 0) {
    for (j in seq_len(p_linear)) {
      x_train[[j]] <- x_linear[, j]
    }
  }

  x_train[[p_linear + 1]] <- z

  inputs <- outputs <- list()

  if (p_linear > 0) {
    for (j in seq_len(p_linear)) {
      inputs[[j]] <- layer_input(
        shape = shape(1),
        name = paste0("linear_input_", j)
      )

      outputs[[j]] <- layer_dense(
        inputs[[j]],
        units = 1,
        use_bias = FALSE,
        activation = NULL,
        kernel_initializer = "normal",
        name = paste0("linear_layer_", j)
      )
    }
  }

  input_z <- layer_input(shape = shape(r), name = "z_input")
  z_current <- input_z

  for (h in seq_len(hidden_length)) {
    z_current <- layer_dense(
      z_current,
      units = widths[h],
      activation = "relu",
      kernel_constraint = constraint_maxnorm(max_value = 1, axis = 0),
      name = paste0("hidden_layer_", h)
    )

    z_current <- layer_dropout(
      z_current,
      rate = dropout,
      name = paste0("dropout_layer_", h)
    )
  }

  z_output <- layer_dense(z_current, units = 1, name = "dnn_output")

  inputs[[p_linear + 1]] <- input_z
  outputs[[p_linear + 1]] <- z_output

  if (length(outputs) == 1) {
    final_output <- outputs[[1]]
  } else {
    final_output <- layer_add(outputs)
  }

  model <- keras_model(inputs = inputs, outputs = final_output)

  if (loss_type == "quantile") {
    model %>% compile(
      loss = function(y_true, y_pred) tilted_loss(tau, y_true, y_pred),
      optimizer = optimizer_adam(),
      metrics = c("mse")
    )
  } else {
    model %>% compile(
      loss = "mse",
      optimizer = optimizer_adam(),
      metrics = c("mse")
    )
  }

  history <- model %>% fit(
    x = x_train,
    y = y,
    epochs = epochs,
    batch_size = batch_size,
    verbose = verbose
  )

  y_pred <- as.numeric(predict(model, x_train))

  list(
    model = model,
    history = history,
    y_pred = y_pred,
    x_train = x_train
  )
}

# ------------------------------------------------------------
# Core rank score test.
#
# Fits the null-restricted quantile model:
#   y = x_nuisance %*% eta + f(z) + error,
# excluding x_interest.
#
# Then projects each column of x_interest onto (x_nuisance, z)
# using least squares and forms:
#   T_n = S_n^T V_n^{-1} S_n.
# ------------------------------------------------------------
rank_score_core_plqr <- function(
    y,
    x_interest,
    z,
    x_nuisance = NULL,
    tau = 0.5,
    hidden_length = 2,
    width = 100,
    dropout = 0.5,
    epochs = 1000,
    batch_size = 4,
    verbose = 1,
    return_fitted = FALSE
) {

  x_interest <- .rank_as_matrix(x_interest, "x_interest")
  z <- .rank_as_matrix(z, "z")
  x_nuisance <- .rank_as_matrix(x_nuisance, "x_nuisance")
  y <- as.numeric(y)

  n <- length(y)
  q_test <- ncol(x_interest)

  if (nrow(x_interest) != n || nrow(z) != n) {
    stop("x_interest, z, and y must have the same number of observations.")
  }
  if (!is.null(x_nuisance) && nrow(x_nuisance) != n) {
    stop("x_nuisance and y must have the same number of observations.")
  }

  # Null-restricted quantile fit.
  null_fit <- fit_unpenalized_pl_dnn(
    x_linear = x_nuisance,
    z = z,
    y = y,
    loss_type = "quantile",
    tau = tau,
    hidden_length = hidden_length,
    width = width,
    dropout = dropout,
    epochs = epochs,
    batch_size = batch_size,
    verbose = verbose
  )

  # Residual under H0: epsilon = y - fitted value.
  epsilon_hat <- y - null_fit$y_pred
  rank_weight <- tau - as.numeric(epsilon_hat < 0)

  # Least-squares projection residuals d_i.
  D <- matrix(NA_real_, nrow = n, ncol = q_test)
  projection_fits <- vector("list", q_test)

  for (j in seq_len(q_test)) {
    proj_fit_j <- fit_unpenalized_pl_dnn(
      x_linear = x_nuisance,
      z = z,
      y = x_interest[, j],
      loss_type = "mse",
      tau = tau,
      hidden_length = hidden_length,
      width = width,
      dropout = dropout,
      epochs = epochs,
      batch_size = batch_size,
      verbose = verbose
    )

    D[, j] <- x_interest[, j] - proj_fit_j$y_pred
    projection_fits[[j]] <- proj_fit_j
  }

  S_n <- as.numeric(crossprod(D, rank_weight) / sqrt(n))
  V_n <- tau * (1 - tau) * crossprod(D) / n

  V_inv <- tryCatch(
    solve(V_n),
    error = function(e) {
      if (!requireNamespace("MASS", quietly = TRUE)) {
        stop("V_n is singular and package MASS is needed for ginv().")
      }
      MASS::ginv(V_n)
    }
  )

  T_n <- as.numeric(t(S_n) %*% V_inv %*% S_n)
  p_value <- 1 - pchisq(T_n, df = q_test)

  out <- list(
    stat = T_n,
    p_value = p_value,
    df = q_test,
    S_n = S_n,
    V_n = V_n,
    D = D,
    epsilon_hat = epsilon_hat
  )

  if (return_fitted) {
    out$null_fit <- null_fit
    out$projection_fits <- projection_fits
  }

  out
}

# ------------------------------------------------------------
# Public wrapper after variable selection.
#
# target_from = "select": test columns of x_select.
# target_from = "keep"  : test columns of x_keep. This is the
#                         closest analog of the old direct-effect test.
# ------------------------------------------------------------
rank_score_test_plqr_dnn <- function(
    y,
    x_select,
    z,
    x_keep = NULL,
    selected_fit,
    target_from = c("select", "keep"),
    target_index = 1,
    tau = 0.5,
    hidden_length = NULL,
    width = NULL,
    dropout = NULL,
    epochs = 1000,
    batch_size = 4,
    verbose = 1,
    return_fitted = FALSE
) {

  target_from <- match.arg(target_from)

  x_select <- .rank_as_matrix(x_select, "x_select")
  z <- .rank_as_matrix(z, "z")
  x_keep <- .rank_as_matrix(x_keep, "x_keep")
  y <- as.numeric(y)

  n <- length(y)
  if (nrow(x_select) != n || nrow(z) != n) {
    stop("x_select, z, and y must have the same number of observations.")
  }
  if (!is.null(x_keep) && nrow(x_keep) != n) {
    stop("x_keep and y must have the same number of observations.")
  }

  # Use final architecture if not explicitly supplied.
  if (is.null(hidden_length)) hidden_length <- selected_fit$hidden_length
  if (is.null(width)) width <- selected_fit$width
  if (is.null(dropout)) dropout <- selected_fit$dropout

  if (is.null(hidden_length) || is.null(width) || is.null(dropout)) {
    stop("hidden_length, width, and dropout must be supplied or stored in selected_fit.")
  }

  # Inherit or reconstruct active set using the same rule as estimation.
  active_set <- selected_fit$active_set
  selection_rule <- selected_fit$selection_rule

  if (is.null(active_set)) {
    if (is.null(selected_fit$alpha_hat)) {
      stop("selected_fit must contain active_set or alpha_hat.")
    }

    if (hidden_length == 1) {
      active_set <- which(selected_fit$alpha_hat != 0)
      selection_rule <- "exact_nonzero_for_single_hidden_layer"
    } else {
      threshold_use <- selected_fit$selection_threshold
      if (is.null(threshold_use)) threshold_use <- 1e-2
      active_set <- which(abs(selected_fit$alpha_hat) >= threshold_use)
      selection_rule <- "threshold_for_deep_network"
    }
  }

  if (is.null(selection_rule)) selection_rule <- "inherited_from_selected_fit"

  if (target_from == "select") {
    if (!all(target_index %in% seq_len(ncol(x_select)))) {
      stop("target_index is outside the columns of x_select.")
    }

    if (!all(target_index %in% active_set)) {
      warning("Some target_index values are not in selected_fit$active_set. ",
              "The rank score test is usually intended for active selected variables.")
    }

    x_interest <- x_select[, target_index, drop = FALSE]
    nuisance_select_index <- setdiff(active_set, target_index)

    x_nuisance_select <- NULL
    if (length(nuisance_select_index) > 0) {
      x_nuisance_select <- x_select[, nuisance_select_index, drop = FALSE]
    }

    x_nuisance <- .rank_cbind_nonnull(x_nuisance_select, x_keep)
  }

  if (target_from == "keep") {
    if (is.null(x_keep)) stop("target_from = 'keep' requires x_keep.")
    if (!all(target_index %in% seq_len(ncol(x_keep)))) {
      stop("target_index is outside the columns of x_keep.")
    }

    x_interest <- x_keep[, target_index, drop = FALSE]

    keep_nuisance_index <- setdiff(seq_len(ncol(x_keep)), target_index)

    x_nuisance_select <- NULL
    if (length(active_set) > 0) {
      x_nuisance_select <- x_select[, active_set, drop = FALSE]
    }

    x_nuisance_keep <- NULL
    if (length(keep_nuisance_index) > 0) {
      x_nuisance_keep <- x_keep[, keep_nuisance_index, drop = FALSE]
    }

    x_nuisance <- .rank_cbind_nonnull(x_nuisance_select, x_nuisance_keep)
  }

  test_result <- rank_score_core_plqr(
    y = y,
    x_interest = x_interest,
    z = z,
    x_nuisance = x_nuisance,
    tau = tau,
    hidden_length = hidden_length,
    width = width,
    dropout = dropout,
    epochs = epochs,
    batch_size = batch_size,
    verbose = verbose,
    return_fitted = return_fitted
  )

  test_result$target_from <- target_from
  test_result$target_index <- target_index
  test_result$active_set <- active_set
  test_result$selection_rule <- selection_rule

  test_result
}
