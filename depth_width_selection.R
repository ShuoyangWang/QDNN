# ============================================================
# depth_width_selection.R
#
# Depth-and-width selection for SCAD-penalized partially linear
# quantile regression.
#
# Requires estimation.R to be sourced first because
# this wrapper calls fit_scad_plqr_dnn(), predict_plqr_total(), and
# predict_plqr_dnn_component().
#
# Current notation:
#   x_select : selected/scannable linear covariates subject to SCAD.
#   x_keep   : optional always-kept, unpenalized linear covariates.
#   z        : nonlinear covariates.
# ============================================================


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

.dw_as_matrix <- function(x, name) {
  if (is.null(x)) return(NULL)
  if (is.null(dim(x))) x <- matrix(x, ncol = 1)
  x <- as.matrix(x)
  if (!is.numeric(x)) stop(name, " must be numeric.")
  x
}

.dw_prepare_widths <- function(width, hidden_length) {
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

.dw_load_scad <- function(scad_dir, scad_id) {
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

.dw_make_linear_input_list <- function(x_select, z, x_keep = NULL) {
  x_select <- .dw_as_matrix(x_select, "x_select")
  z <- .dw_as_matrix(z, "z")
  x_keep <- .dw_as_matrix(x_keep, "x_keep")

  p_select <- ncol(x_select)
  p_keep <- ifelse(is.null(x_keep), 0, ncol(x_keep))
  r <- ncol(z)

  x_list <- vector("list", p_select + p_keep + r)

  for (j in seq_len(p_select)) {
    x_list[[j]] <- x_select[, j]
  }

  if (p_keep > 0) {
    for (j in seq_len(p_keep)) {
      x_list[[p_select + j]] <- x_keep[, j]
    }
  }

  for (j in seq_len(r)) {
    x_list[[p_select + p_keep + j]] <- z[, j]
  }

  x_list
}

.dw_subset_keep <- function(x_keep, ids) {
  if (is.null(x_keep)) return(NULL)
  x_keep[ids, , drop = FALSE]
}

# ------------------------------------------------------------
# Preliminary linear SCAD quantile regression.
#
# Fits:
#   y = x_select %*% alpha + x_keep %*% gamma + z %*% theta + error
# with SCAD penalty only on alpha.
#
# This Keras/Adam implementation is used for consistency with the
# nonlinear estimator. Users may replace this helper with a
# quantreg::rq-based linear quantile-regression routine, for example
# within a local-linear-approximation or coordinate-descent implementation
# for SCAD. Note that quantreg::rq() itself is unpenalized unless
# additional penalty machinery is supplied.
# ------------------------------------------------------------
fit_scad_linear_qr <- function(
    x_select,
    z,
    y,
    x_keep = NULL,
    tau = 0.5,
    scad_dir = "~/QMDNN/auxiliary",
    scad_ids = 1:8,
    epochs = 1000,
    batch_size = 4,
    selection_threshold = 1e-2,
    verbose = 1
) {

  library(keras)
  library(tensorflow)

  x_select <- .dw_as_matrix(x_select, "x_select")
  z <- .dw_as_matrix(z, "z")
  x_keep <- .dw_as_matrix(x_keep, "x_keep")
  y <- as.numeric(y)

  n <- length(y)
  p_select <- ncol(x_select)
  p_keep <- ifelse(is.null(x_keep), 0, ncol(x_keep))
  r <- ncol(z)

  if (nrow(x_select) != n || nrow(z) != n) {
    stop("x_select, z, and y must have the same number of observations.")
  }
  if (!is.null(x_keep) && nrow(x_keep) != n) {
    stop("x_keep and y must have the same number of observations.")
  }

  tilted_loss <- function(qtau, y_true, y_pred) {
    e <- y_true - y_pred
    k_mean(k_maximum(qtau * e, (qtau - 1) * e), axis = 2)
  }

  build_linear_model <- function(scad_fun) {
    inputs <- outputs <- list()

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

    for (j in seq_len(r)) {
      idx <- p_select + p_keep + j
      inputs[[idx]] <- layer_input(
        shape = shape(1),
        name = paste0("z_linear_input_", j)
      )

      outputs[[idx]] <- layer_dense(
        inputs[[idx]],
        units = 1,
        use_bias = FALSE,
        activation = NULL,
        kernel_initializer = "normal",
        name = paste0("z_linear_", j)
      )
    }

    final_output <- layer_add(outputs)
    model <- keras_model(inputs = inputs, outputs = final_output)

    model %>% compile(
      loss = function(y_true, y_pred) tilted_loss(tau, y_true, y_pred),
      optimizer = optimizer_adam(),
      metrics = c("mse")
    )

    model
  }

  extract_linear_coefs <- function(model) {
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

    theta_hat <- numeric(r)
    for (j in seq_len(r)) {
      theta_hat[j] <- as.numeric(
        get_weights(get_layer(model, name = paste0("z_linear_", j)))[[1]]
      )
    }

    list(alpha_hat = alpha_hat, gamma_hat = gamma_hat, theta_hat = theta_hat)
  }

  x_train <- .dw_make_linear_input_list(x_select = x_select, z = z, x_keep = x_keep)

  HBIC <- rep(NA_real_, length(scad_ids))
  df <- rep(NA_integer_, length(scad_ids))
  fit_list <- vector("list", length(scad_ids))

  for (k in seq_along(scad_ids)) {
    scad_id <- scad_ids[k]
    scad_fun <- .dw_load_scad(scad_dir = scad_dir, scad_id = scad_id)

    model <- build_linear_model(scad_fun)

    history <- model %>% fit(
      x = x_train,
      y = y,
      epochs = epochs,
      batch_size = batch_size,
      verbose = verbose
    )

    y_pred <- as.numeric(predict(model, x_train))
    coefs <- extract_linear_coefs(model)

    df[k] <- sum(abs(coefs$alpha_hat) >= selection_threshold)
    rho_val <- max(plqr_rho_sum(y - y_pred, tau), .Machine$double.eps)

    HBIC[k] <- log10(rho_val) +
      df[k] * log10(log10(n)) * log10(p_select + p_keep + r) / n

    fit_list[[k]] <- list(
      model = model,
      history = history,
      alpha_hat = coefs$alpha_hat,
      gamma_hat = coefs$gamma_hat,
      theta_hat = coefs$theta_hat,
      y_pred = y_pred
    )

    gc()
  }

  best_pos <- which(HBIC == min(HBIC))[1]
  best_fit <- fit_list[[best_pos]]

  list(
    alpha_hat = best_fit$alpha_hat,
    gamma_hat = best_fit$gamma_hat,
    theta_hat = best_fit$theta_hat,
    model = best_fit$model,
    history = best_fit$history,
    tuning_best = scad_ids[best_pos],
    HBIC = HBIC,
    df = df
  )
}

# ------------------------------------------------------------
# Main depth-and-width selection wrapper.
# ------------------------------------------------------------
depth_width_select_scad_plqr <- function(
    x_select,
    z,
    y,
    x_keep = NULL,
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
    dropout_deep = 0.5,
    seed = 123,
    verbose = 1,
    use_conda_python = TRUE,
    return_all_fits = FALSE
) {

  library(keras)
  library(tensorflow)
  library(reticulate)

  if (!exists("fit_scad_plqr_dnn", mode = "function")) {
    stop("Please source estimation_scad_plqr_dnn.R before this file.")
  }

  if (!exists("predict_plqr_total", mode = "function") ||
      !exists("predict_plqr_dnn_component", mode = "function")) {
    stop("Prediction helpers are missing. Source estimation_scad_plqr_dnn.R first.")
  }

  if (use_conda_python) {
    use_python(paste(Sys.getenv("CONDA_PREFIX"), "bin/python", sep = "/"))
  }

  x_select <- .dw_as_matrix(x_select, "x_select")
  z <- .dw_as_matrix(z, "z")
  x_keep <- .dw_as_matrix(x_keep, "x_keep")
  y <- as.numeric(y)

  n <- length(y)

  if (nrow(x_select) != n || nrow(z) != n) {
    stop("x_select, z, and y must have the same number of observations.")
  }
  if (!is.null(x_keep) && nrow(x_keep) != n) {
    stop("x_keep and y must have the same number of observations.")
  }

  if (abs(train_prop + val_prop + test_prop - 1) > 1e-8) {
    stop("train_prop + val_prop + test_prop must equal 1.")
  }

  if (any(L_deep <= 1)) {
    stop("L_deep should only contain depths larger than 1.")
  }

  N_max <- max(N_width)

  set.seed(seed)
  id_all <- sample(seq_len(n))

  n_train <- floor(train_prop * n)
  n_val <- floor(val_prop * n)

  if (n_train < 1 || n_val < 1 || n_train + n_val >= n) {
    stop("The requested train/validation/test split leaves an empty subset.")
  }

  id_tr <- id_all[seq_len(n_train)]
  id_val <- id_all[(n_train + 1):(n_train + n_val)]
  id_te <- id_all[(n_train + n_val + 1):n]

  x_select_tr <- x_select[id_tr, , drop = FALSE]
  x_select_val <- x_select[id_val, , drop = FALSE]
  x_select_te <- x_select[id_te, , drop = FALSE]

  x_keep_tr <- .dw_subset_keep(x_keep, id_tr)
  x_keep_val <- .dw_subset_keep(x_keep, id_val)
  x_keep_te <- .dw_subset_keep(x_keep, id_te)

  z_tr <- z[id_tr, , drop = FALSE]
  z_val <- z[id_val, , drop = FALSE]
  z_te <- z[id_te, , drop = FALSE]

  y_tr <- y[id_tr]
  y_val <- y[id_val]
  y_te <- y[id_te]

  # ----------------------------------------------------------
  # Linear estimates on training set.
  # ----------------------------------------------------------
  lin_fit <- fit_scad_linear_qr(
    x_select = x_select_tr,
    x_keep = x_keep_tr,
    z = z_tr,
    y = y_tr,
    tau = tau,
    scad_dir = scad_dir,
    scad_ids = scad_ids,
    epochs = epochs,
    batch_size = batch_size,
    selection_threshold = selection_threshold,
    verbose = verbose
  )

  x_alpha_lin_val <- as.numeric(x_select_val %*% lin_fit$alpha_hat)
  z_theta_lin_val <- as.numeric(z_val %*% lin_fit$theta_hat)

  # ----------------------------------------------------------
  # Shallow L = 1 model using N_max on training set.
  # ----------------------------------------------------------
  shallow_Nmax_fit <- fit_scad_plqr_dnn(
    x_select = x_select_tr,
    x_keep = x_keep_tr,
    z = z_tr,
    y = y_tr,
    tau = tau,
    scad_dir = scad_dir,
    scad_ids = scad_ids,
    hidden_length = 1,
    width = N_max,
    dropout = dropout_shallow,
    epochs = epochs,
    batch_size = batch_size,
    selection_threshold = selection_threshold,
    verbose = verbose,
    use_conda_python = FALSE,
    return_model = TRUE
  )

  x_alpha_shallow_val <- as.numeric(x_select_val %*% shallow_Nmax_fit$alpha_hat)
  f_shallow_val <- predict_plqr_dnn_component(shallow_Nmax_fit, z_val)

  # ----------------------------------------------------------
  # Validation discrepancies D1 and D2.
  # ----------------------------------------------------------
  delta1 <- abs(x_alpha_lin_val - x_alpha_shallow_val)
  delta2 <- abs(z_theta_lin_val - f_shallow_val)

  D1 <- mean(delta1)
  D2 <- mean(delta2)

  # ----------------------------------------------------------
  # Architecture search by validation quantile loss.
  # ----------------------------------------------------------
  architecture_results <- data.frame(
    L = integer(0),
    N = numeric(0),
    validation_loss = numeric(0)
  )

  fit_cache <- list()
  fit_cache[[paste0("L1_N", N_max)]] <- shallow_Nmax_fit

  get_arch_fit <- function(L, N) {
    key <- paste0("L", L, "_N", N)

    if (!is.null(fit_cache[[key]])) {
      return(fit_cache[[key]])
    }

    dropout_use <- ifelse(L == 1, dropout_shallow, dropout_deep)

    fit <- fit_scad_plqr_dnn(
      x_select = x_select_tr,
      x_keep = x_keep_tr,
      z = z_tr,
      y = y_tr,
      tau = tau,
      scad_dir = scad_dir,
      scad_ids = scad_ids,
      hidden_length = L,
      width = N,
      dropout = dropout_use,
      epochs = epochs,
      batch_size = batch_size,
      selection_threshold = selection_threshold,
      verbose = verbose,
      use_conda_python = FALSE,
      return_model = TRUE
    )

    fit_cache[[key]] <<- fit
    fit
  }

  if (D1 >= kappa * D2) {
    architecture_branch <- "shallow"

    for (N in N_width) {
      fit_LN <- get_arch_fit(L = 1, N = N)
      y_hat_val <- predict_plqr_total(
        fit = fit_LN,
        x_select = x_select_val,
        x_keep = x_keep_val,
        z = z_val
      )

      VL <- plqr_rho_mean(y_val - y_hat_val, tau)

      architecture_results <- rbind(
        architecture_results,
        data.frame(L = 1, N = N, validation_loss = VL)
      )
    }
  } else {
    architecture_branch <- "deep"

    for (L in L_deep) {
      for (N in N_width) {
        fit_LN <- get_arch_fit(L = L, N = N)
        y_hat_val <- predict_plqr_total(
          fit = fit_LN,
          x_select = x_select_val,
          x_keep = x_keep_val,
          z = z_val
        )

        VL <- plqr_rho_mean(y_val - y_hat_val, tau)

        architecture_results <- rbind(
          architecture_results,
          data.frame(L = L, N = N, validation_loss = VL)
        )
      }
    }
  }

  best_row <- which.min(architecture_results$validation_loss)
  L_star <- architecture_results$L[best_row]
  N_star <- architecture_results$N[best_row]

  # ----------------------------------------------------------
  # Final fit on training + validation set.
  # ----------------------------------------------------------
  id_trval <- c(id_tr, id_val)

  x_select_trval <- x_select[id_trval, , drop = FALSE]
  x_keep_trval <- .dw_subset_keep(x_keep, id_trval)
  z_trval <- z[id_trval, , drop = FALSE]
  y_trval <- y[id_trval]

  dropout_final <- ifelse(L_star == 1, dropout_shallow, dropout_deep)

  final_fit <- fit_scad_plqr_dnn(
    x_select = x_select_trval,
    x_keep = x_keep_trval,
    z = z_trval,
    y = y_trval,
    tau = tau,
    scad_dir = scad_dir,
    scad_ids = scad_ids,
    hidden_length = L_star,
    width = N_star,
    dropout = dropout_final,
    epochs = epochs,
    batch_size = batch_size,
    selection_threshold = selection_threshold,
    verbose = verbose,
    use_conda_python = FALSE,
    return_model = TRUE
  )

  # Defensive enforcement of the depth-dependent active-set rule.
  if (L_star == 1) {
    active_set_final <- which(final_fit$alpha_hat != 0)
    selection_rule <- "exact_nonzero_for_single_hidden_layer"
  } else {
    active_set_final <- which(abs(final_fit$alpha_hat) >= selection_threshold)
    selection_rule <- "threshold_for_deep_network"
  }

  final_fit$active_set <- active_set_final
  final_fit$selection_rule <- selection_rule

  y_hat_test <- predict_plqr_total(
    fit = final_fit,
    x_select = x_select_te,
    x_keep = x_keep_te,
    z = z_te
  )

  test_loss <- plqr_rho_mean(y_te - y_hat_test, tau)

  out <- list(
    alpha_hat = final_fit$alpha_hat,
    gamma_hat = final_fit$gamma_hat,
    W_hat = get_weights(final_fit$model),
    active_set = final_fit$active_set,

    L_star = L_star,
    N_star = N_star,
    selected_architecture = c(L = L_star, N = N_star),
    architecture_branch = architecture_branch,
    architecture_results = architecture_results,

    D1 = D1,
    D2 = D2,
    kappa = kappa,
    selection_rule = final_fit$selection_rule,

    final_fit = final_fit,
    linear_fit = lin_fit,
    shallow_Nmax_fit = shallow_Nmax_fit,

    test_loss = test_loss,
    y_hat_test = y_hat_test,

    split = list(
      train = id_tr,
      validation = id_val,
      test = id_te,
      train_validation = id_trval
    ),

    settings = list(
      tau = tau,
      L_deep = L_deep,
      N_width = N_width,
      N_max = N_max,
      selection_threshold = selection_threshold,
      epochs = epochs,
      batch_size = batch_size,
      dropout_shallow = dropout_shallow,
      dropout_deep = dropout_deep,
      scad_ids = scad_ids
    )
  )

  if (return_all_fits) {
    out$fit_cache <- fit_cache
  }

  out
}
