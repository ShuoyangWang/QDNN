# ============================================================
# estimation_scad_plqr_dnn.R
#
# Algorithm 1 computational implementation:
# Adam for SCAD-penalized partially linear quantile regression
# with a neural-network nonlinear component.
#
# Model:
#   y = x_select %*% alpha + x_keep %*% gamma + f_W(z) + error
#
# Notation:
#   x_select : linear covariates subject to SCAD selection.
#              In the old mediation code, this was M.
#   x_keep   : optional linear covariates always kept in the model.
#              In the old mediation code, this was x.
#   z        : nonlinear covariates modeled by the DNN.
#
# Important selection rule:
#   If hidden_length == 1, use exact nonzero coefficients:
#       A_hat = {j: alpha_hat_j != 0}.
#   If hidden_length > 1, use thresholding:
#       A_hat = {j: |alpha_hat_j| >= selection_threshold}.
# ============================================================


# ------------------------------------------------------------
# Internal utilities
# ------------------------------------------------------------

.plqr_as_matrix <- function(x, name) {
  if (is.null(x)) return(NULL)
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  if (!is.numeric(x)) stop(name, " must be numeric.")
  x
}

plqr_rho_values <- function(resid, tau) {
  ifelse(resid < 0, (tau - 1) * resid, tau * resid)
}

plqr_rho_sum <- function(resid, tau) {
  sum(plqr_rho_values(resid, tau))
}

plqr_rho_mean <- function(resid, tau) {
  mean(plqr_rho_values(resid, tau))
}

.plqr_prepare_widths <- function(width, hidden_length) {
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

.plqr_load_scad <- function(scad_dir, scad_id) {
  scad_file <- file.path(scad_dir, paste0("scad", scad_id, ".R"))

  if (!file.exists(scad_file)) {
    stop("Cannot find SCAD file: ", scad_file,
         "\nRun SCAD.R first or set scad_dir correctly.")
  }

  env_scad <- new.env(parent = globalenv())
  source(scad_file, local = env_scad)

  if (!exists("scad", envir = env_scad, inherits = FALSE)) {
    stop("The file ", scad_file, " did not define a function named scad.")
  }

  get("scad", envir = env_scad)
}

# ------------------------------------------------------------
# Public helper: construct Keras input list in the same order as
# fit_scad_plqr_dnn().
# ------------------------------------------------------------
make_plqr_input_list <- function(x_select, z, x_keep = NULL) {
  x_select <- .plqr_as_matrix(x_select, "x_select")
  z <- .plqr_as_matrix(z, "z")
  x_keep <- .plqr_as_matrix(x_keep, "x_keep")

  n <- nrow(x_select)
  if (nrow(z) != n) stop("x_select and z must have the same number of rows.")
  if (!is.null(x_keep) && nrow(x_keep) != n) {
    stop("x_select and x_keep must have the same number of rows.")
  }

  p_select <- ncol(x_select)
  p_keep <- ifelse(is.null(x_keep), 0, ncol(x_keep))

  x_list <- vector("list", p_select + p_keep + 1)

  for (j in seq_len(p_select)) {
    x_list[[j]] <- x_select[, j]
  }

  if (p_keep > 0) {
    for (j in seq_len(p_keep)) {
      x_list[[p_select + j]] <- x_keep[, j]
    }
  }

  x_list[[p_select + p_keep + 1]] <- z
  x_list
}

# ------------------------------------------------------------
# Public helper: predict the full partially linear DNN fitted value.
# ------------------------------------------------------------
predict_plqr_total <- function(fit, x_select, z, x_keep = NULL) {
  if (is.null(fit$model)) {
    stop("fit must contain a Keras model. Use return_model = TRUE.")
  }

  x_list <- make_plqr_input_list(
    x_select = x_select,
    z = z,
    x_keep = x_keep
  )

  as.numeric(predict(fit$model, x_list))
}

# ------------------------------------------------------------
# Public helper: predict only the nonlinear DNN component f_W(z).
# This requires the layer names used in fit_scad_plqr_dnn().
# ------------------------------------------------------------
predict_plqr_dnn_component <- function(fit, z) {
  if (is.null(fit$model)) {
    stop("fit must contain a Keras model. Use return_model = TRUE.")
  }

  z <- .plqr_as_matrix(z, "z")

  z_input <- get_layer(fit$model, name = "z_input")$input
  z_output <- get_layer(fit$model, name = "dnn_output")$output

  dnn_model <- keras_model(inputs = z_input, outputs = z_output)
  as.numeric(predict(dnn_model, z))
}

# ------------------------------------------------------------
# Main estimation function: Algorithm 1 computational version.
# ------------------------------------------------------------
fit_scad_plqr_dnn <- function(
    x_select,
    z,
    y,
    x_keep = NULL,
    tau = 0.5,
    scad_dir = "~/QMDNN/auxiliary",
    scad_ids = 1:8,
    hidden_length = 2,
    width = 100,
    dropout = 0.5,
    epochs = 1000,
    batch_size = 4,
    selection_threshold = 1e-2,
    verbose = 1,
    use_conda_python = TRUE,
    return_model = FALSE
) {

  library(keras)
  library(tensorflow)
  library(reticulate)

  if (use_conda_python) {
    use_python(paste(Sys.getenv("CONDA_PREFIX"), "bin/python", sep = "/"))
  }

  x_select <- .plqr_as_matrix(x_select, "x_select")
  z <- .plqr_as_matrix(z, "z")
  x_keep <- .plqr_as_matrix(x_keep, "x_keep")
  y <- as.numeric(y)

  n <- length(y)
  p_select <- ncol(x_select)
  r <- ncol(z)
  p_keep <- ifelse(is.null(x_keep), 0, ncol(x_keep))

  if (nrow(x_select) != n || nrow(z) != n) {
    stop("x_select, z, and y must have the same number of observations.")
  }
  if (!is.null(x_keep) && nrow(x_keep) != n) {
    stop("x_keep and y must have the same number of observations.")
  }

  widths <- .plqr_prepare_widths(width, hidden_length)

  tilted_loss <- function(qtau, y_true, y_pred) {
    e <- y_true - y_pred
    k_mean(k_maximum(qtau * e, (qtau - 1) * e), axis = 2)
  }

  x_train <- make_plqr_input_list(x_select = x_select, z = z, x_keep = x_keep)
  y_train <- y

  # ----------------------------------------------------------
  # Active-set rule used both for HBIC degrees of freedom and
  # for the final returned active set.
  # ----------------------------------------------------------
  get_active_set <- function(alpha_hat) {
    if (hidden_length == 1) {
      active_set <- which(alpha_hat != 0)
      selection_rule <- "exact_nonzero_for_single_hidden_layer"
    } else {
      active_set <- which(abs(alpha_hat) >= selection_threshold)
      selection_rule <- "threshold_for_deep_network"
    }

    list(
      active_set = active_set,
      df = length(active_set),
      selection_rule = selection_rule
    )
  }

  build_model <- function(scad_fun) {

    inputs <- outputs <- list()

    # SCAD-penalized selected linear covariates.
    for (j in seq_len(p_select)) {
      inputs[[j]] <- layer_input(
        shape = shape(1),
        name = paste0("x_select_input_", j)
      )

      outputs[[j]] <- layer_dense(
        inputs[[j]],
        units = 1,
        use_bias = FALSE,
        activation = NULL,
        kernel_regularizer = scad_fun,
        kernel_initializer = "zero",
        name = paste0("x_select_linear_", j)
      )
    }

    # Unpenalized linear covariates that are always kept.
    if (p_keep > 0) {
      for (j in seq_len(p_keep)) {
        idx <- p_select + j
        inputs[[idx]] <- layer_input(
          shape = shape(1),
          name = paste0("x_keep_input_", j)
        )

        outputs[[idx]] <- layer_dense(
          inputs[[idx]],
          units = 1,
          use_bias = FALSE,
          activation = NULL,
          kernel_initializer = "normal",
          name = paste0("x_keep_linear_", j)
        )
      }
    }

    # Nonlinear DNN component f_W(z).
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

    inputs[[p_select + p_keep + 1]] <- input_z
    outputs[[p_select + p_keep + 1]] <- z_output

    final_output <- layer_add(outputs)
    model <- keras_model(inputs = inputs, outputs = final_output)

    model %>% compile(
      loss = function(y_true, y_pred) tilted_loss(tau, y_true, y_pred),
      optimizer = optimizer_adam(),
      metrics = c("mse")
    )

    model
  }

  extract_linear_coefficients <- function(model) {
    alpha_hat <- numeric(p_select)
    for (j in seq_len(p_select)) {
      alpha_hat[j] <- as.numeric(
        get_weights(get_layer(model, name = paste0("x_select_linear_", j)))[[1]]
      )
    }

    gamma_hat <- numeric(p_keep)
    if (p_keep > 0) {
      for (j in seq_len(p_keep)) {
        gamma_hat[j] <- as.numeric(
          get_weights(get_layer(model, name = paste0("x_keep_linear_", j)))[[1]]
        )
      }
    }

    list(alpha_hat = alpha_hat, gamma_hat = gamma_hat)
  }

  fit_one_lambda <- function(scad_id, keep_model = FALSE) {
    scad_fun <- .plqr_load_scad(scad_dir = scad_dir, scad_id = scad_id)
    model <- build_model(scad_fun)

    history <- model %>% fit(
      x = x_train,
      y = y_train,
      epochs = epochs,
      batch_size = batch_size,
      verbose = verbose
    )

    coef_info <- extract_linear_coefficients(model)
    y_pred <- as.numeric(predict(model, x_train))

    active_info <- get_active_set(coef_info$alpha_hat)
    df <- active_info$df

    rho_val <- max(plqr_rho_sum(y_train - y_pred, tau), .Machine$double.eps)
    HBIC <- log10(rho_val) +
      df * log10(log10(n)) * log10(p_select + p_keep) / n

    out <- list(
      scad_id = scad_id,
      HBIC = HBIC,
      df = df,
      alpha_hat = coef_info$alpha_hat,
      gamma_hat = coef_info$gamma_hat,
      active_set = active_info$active_set,
      selection_rule = active_info$selection_rule,
      y_pred = y_pred
    )

    if (keep_model) {
      out$model <- model
      out$history <- history
    }

    out
  }

  t1 <- Sys.time()

  HBIC <- rep(NA_real_, length(scad_ids))
  df <- rep(NA_integer_, length(scad_ids))

  for (k in seq_along(scad_ids)) {
    fit_k <- fit_one_lambda(scad_id = scad_ids[k], keep_model = FALSE)
    HBIC[k] <- fit_k$HBIC
    df[k] <- fit_k$df
    gc()
  }

  best_pos <- which(HBIC == min(HBIC))[1]
  tuning_best <- scad_ids[best_pos]

  final_fit <- fit_one_lambda(scad_id = tuning_best, keep_model = return_model)

  runtime <- Sys.time() - t1

  result <- list(
    alpha_hat = final_fit$alpha_hat,
    gamma_hat = final_fit$gamma_hat,
    active_set = final_fit$active_set,
    selection_rule = final_fit$selection_rule,
    y_pred = final_fit$y_pred,

    tuning_best = tuning_best,
    HBIC = HBIC,
    df = df,

    tau = tau,
    p_select = p_select,
    p_keep = p_keep,
    r = r,
    hidden_length = hidden_length,
    width = widths,
    dropout = dropout,
    selection_threshold = selection_threshold,
    runtime = runtime
  )

  if (return_model) {
    result$model <- final_fit$model
    result$history <- final_fit$history
  }

  result
}
