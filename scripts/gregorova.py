# src/pyselect/gregorova.py
"""Code from Gregorova *et al* implementing the Sparse Random 
Fourier Features (SRF) method.
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

params = yaml.safe_load(open("../gregorova-params.yaml"))
max_iter_gamma = params["max_iter_gamma"]
max_iter_srf = params["max_iter_srf"]
num_avg_samples = params["num_avg_samples"]
update_threshold = params["update_threshold"]
out_features = params["out_features"]
kernel_param = params["kernel_param"]
lambda_min = params["lambda_min"]
lambda_max = params["lambda_max"]
lambda_step = params["lambda_step"]


def gamma_fista_loss(g_temp, X, B, y, a, epsilon, n):
    omega = epsilon.mm(torch.diag(g_temp))
    Z = torch.cos(X.mm(torch.transpose(omega, 1, 0)) + B)
    diff = (y - Z.mm(a)).squeeze()
    return 0.5 / n * diff.dot(diff)


def simplex_project_vectorised(in_vec, simplex_size=1):
    n_elements = len(in_vec)

    sorted_vec, _ = torch.sort(in_vec, descending=True)
    list_idx = torch.arange(1, n_elements)
    tmpsum = torch.cumsum(sorted_vec, dim=0)
    tmpmax = (torch.squeeze(tmpsum.data[: n_elements - 1]) - simplex_size) / list_idx
    tcheck = torch.ge(tmpmax, torch.squeeze(sorted_vec.data[1:n_elements]))

    if torch.sum(tcheck) > 0:
        tmax_ind = torch.min(torch.masked_select(list_idx, tcheck)) - 1
        tmax = tmpmax[int(tmax_ind)]
    else:
        tmax = (tmpsum[n_elements - 1] - simplex_size) / n_elements

    out_vec = torch.max(in_vec - tmax, torch.zeros_like(in_vec))

    return torch.squeeze(out_vec)


def gamma_fista(X, y, init_g, a, epsilon, B, constrain_size):
    n_samples, n_features = X.size()
    obj_history = np.zeros(max_iter_gamma, dtype=np.float32)

    out_g = init_g.clone()
    obj_history[0] = gamma_fista_loss(out_g, X, B, y, a, epsilon, n_samples)

    titer, gY, alpha, beta = (
        1.0,
        out_g.clone(),
        100,
        0.5,
    )  # FISTA update variables? Hard coded.

    for i in range(1, max_iter_gamma):
        epsilon = torch.autograd.Variable(epsilon)
        gY = torch.autograd.Variable(gY, requires_grad=True)
        B = torch.autograd.Variable(B)
        X = torch.autograd.Variable(X)
        y = torch.autograd.Variable(y)
        a = torch.autograd.Variable(a)
        out_g = torch.autograd.Variable(out_g)

        grad_g = torch.autograd.grad(
            gamma_fista_loss(gY, X, B, y, a, epsilon, n_samples), gY
        )[0]

        assert grad_g.size() == gY.size(), "grads [{}] do not match tensor [{}]".format(
            grad_g.size(), gY.size()
        )

        while alpha > update_threshold:
            grad_update = gY - alpha * grad_g

            gNew = simplex_project_vectorised(
                torch.unsqueeze(grad_update, 1), simplex_size=constrain_size
            )

            gDiff = gNew - gY
            objNew = gamma_fista_loss(gNew, X, B, y, a, epsilon, n_samples)
            objY = gamma_fista_loss(gY, X, B, y, a, epsilon, n_samples)
            qual = objY + grad_g.dot(gDiff) + (0.5 / alpha * (gDiff.dot(gDiff)))

            if objNew.item() <= qual.item():
                tNew = 0.5 + np.sqrt(1 + 4.0 * titer * titer) / 2  # eq 4.2
                gY = gNew + ((titer - 1) / tNew * (gNew - out_g))  # eq. 4.3
                titer, out_g = tNew, gNew.clone()  #  and update titer and gamma
                changedG = 1  # updated g
                break
            else:
                changedG = 0  # g same as before
                alpha = alpha * beta

        if changedG:
            loss = gamma_fista_loss(out_g, X, B, y, a, epsilon, n_samples)
            obj_history[i] = loss.detach().item()
        else:
            obj_history[i] = obj_history[i - 1]

        if (
            i > num_avg_samples
            and sum(obj_history[i - (num_avg_samples - 1) : i] - obj_history[i])
            < update_threshold
        ):
            # sanity check if there is no update throughout
            if obj_history[i] - obj_history[0] > update_threshold:
                # print(obj_history)
                print(
                    "gammaFISTA: something fishy obj_history[{}]={} > obj_history[0]={}".format(
                        i, obj_history[i], obj_history[0]
                    )
                )

            break

    return out_g


def loss_function(a, y, Z, reg_parameter, n):
    diff = (y - Z.mm(a)).squeeze()
    a_squeezed = a.squeeze()
    return (0.5 / n) * diff.dot(diff) + (
        0.5 * reg_parameter * a_squeezed.dot(a_squeezed)
    )


def srf_algo(X, y, reg_parameter):
    n_samples, n_features = X.size()
    epsilon = torch.randn(out_features, n_features)
    b = 2 * np.pi * torch.rand((out_features, 1))
    B = torch.ones((n_samples, 1)).mm(torch.transpose(b, 1, 0))
    gamma = torch.ones([n_features]) / kernel_param
    omega = epsilon.mm(torch.diag(gamma.data))

    constrain_size = torch.sum(gamma.data)

    Z = torch.cos(X.mm(torch.transpose(omega, 1, 0)) + B)

    print(Z.size())
    out_features_eye = torch.eye(out_features)

    print(torch.transpose(Z, 1, 0).mm(y).size())
    a = torch.linalg.solve(
        torch.transpose(Z, 1, 0).mm(Z) + n_samples * reg_parameter * out_features_eye,
        torch.transpose(Z, 1, 0).mm(y),
    )

    obj_history = np.zeros(max_iter_srf, dtype=np.float32)
    obj_history[0] = loss_function(a, y, Z, reg_parameter, n_samples)

    for i in range(1, max_iter_srf):
        gamma = gamma_fista(X, y, gamma, a, epsilon, B, constrain_size)

        omega = epsilon.mm(torch.diag(gamma.data))
        Z = torch.cos(X.mm(torch.transpose(omega, 1, 0)) + B)
        a = torch.linalg.solve(
            torch.transpose(Z, 1, 0).mm(Z)
            + n_samples * reg_parameter * out_features_eye,
            torch.transpose(Z, 1, 0).mm(y),
        )

        obj_history[i] = loss_function(a, y, Z, reg_parameter, n_samples)

        # keep track of the objective history
        if obj_history[i] > obj_history[i - 1]:
            print(
                "SRF: something fishy obj_history[{}]={} > obj_history[{}]={}\n".format(
                    i, obj_history[i], i - 1, obj_history[i]
                )
            )

        if (
            i > num_avg_samples
            and sum(obj_history[i - (num_avg_samples - 1) : i] - obj_history[i])
            < update_threshold
        ):
            print(
                "update thresh [{}] satisfied at interval {}, exiting...".format(
                    update_threshold, i
                )
            )
            break

    return {"obj_history": obj_history, "omg": omega, "b": b, "a": a, "gamma": gamma}


def normest(mat):
    _, S, _ = mat.svd()
    return torch.max(S)


def predict_linear(X, y, w, reduce_dim=0):
    preds = X.mm(w)
    error = torch.mean(torch.square(preds - y), reduce_dim)
    return preds, error


def find_min_valid_error(valid_results_map):
    if isinstance(valid_results_map, (list, tuple)):
        valid_errors = np.vstack([vm["error"] for vm in valid_results_map]).reshape(-1)
    else:
        valid_errors = valid_results_map["error"]

    min_idx = np.argmin(valid_errors)
    min_val = valid_errors[min_idx]
    return min_val, min_idx


def srf_run(X_train, y_train, X_val, y_val, X_test, y_test):

    # The maximal eigenvalue of X^T X seems to be important
    # for scaling the choice of lambdas in the validation grid
    sigma = normest(torch.transpose(X_train, 1, 0).mm(X_train))
    print("Estimated sigma: ", sigma)

    range_lambdas = torch.arange(lambda_min, lambda_max, lambda_step)
    lambdas = sigma * torch.pow(10, range_lambdas)

    train_results_map = [{} for _ in range(len(lambdas))]
    valid_results_map = [{} for _ in range(len(lambdas))]

    for i, lambda_val in enumerate(lambdas):
        print(f"SRF : training {i}: with {lambda_val}")
        train_results_map[i] = srf_algo(X_train, y_train, lambda_val)

        ones_mat = torch.ones([X_val.size(0), 1])
        Z_valid = torch.cos(
            X_val.mm(torch.transpose(train_results_map[i]["omg"], 1, 0))
            + ones_mat.mm(torch.transpose(train_results_map[i]["b"], 1, 0))
        )
        _, val_error = predict_linear(Z_valid, y_val, train_results_map[i]["a"])
        valid_results_map[i]["error"] = np.asarray([val_error])

    min_val_error, min_val_idx = find_min_valid_error(valid_results_map)

    test_results_map = {
        "lambda_idx": np.array([min_val_idx]),
        "lambda": np.array([lambdas[min_val_idx]]),
        "gamma": train_results_map[min_val_idx]["gamma"],
        "a": train_results_map[min_val_idx]["a"],
        "b": train_results_map[min_val_idx]["b"],
        "omg": train_results_map[min_val_idx]["omg"],
    }

    ones_mat = torch.ones([X_test.size(0), 1])
    Z_test = torch.cos(
        X_test.mm(torch.transpose(test_results_map["omg"], 1, 0))
        + ones_mat.mm(torch.transpose(test_results_map["b"], 1, 0))
    )
    test_preds, test_error = predict_linear(Z_test, y_test, test_results_map["a"])
    test_results_map["preds"] = test_preds
    test_results_map["error"] = np.asarray([test_error])

    return train_results_map, valid_results_map, test_results_map
