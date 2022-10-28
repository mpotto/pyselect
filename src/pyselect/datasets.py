import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state


# From skglm library.
def make_correlated_data(
    n_samples=100,
    n_features=50,
    n_tasks=1,
    rho=0.6,
    snr=3,
    w_true=None,
    density=0.2,
    X_density=1,
    random_state=None,
):
    r"""Generate a linear regression with correlated design.
    The data are generated according to:
    .. math ::
        y = X w^* + \epsilon
    such that the signal to noise ratio is
    :math:`snr = \frac{||X w^*||}{||\epsilon||}`.
    The generated features have mean 0, variance 1 and the expected correlation
    structure
    .. math ::
        \mathbb E[x_i] = 0~, \quad \mathbb E[x_i^2] = 1  \quad
        and \quad \mathbb E[x_ix_j] = \rho^{|i-j|}
    Parameters
    ----------
    n_samples : int
        Number of samples in the design matrix.
    n_features : int
        Number of features in the design matrix.
    n_tasks : int
        Number of tasks.
    rho : float
        Correlation :math:`\rho` between successive features. The cross
        correlation :math:`C_{i, j}` between feature i and feature j will be
        :math:`\rho^{|i-j|}`. This parameter should be selected in
        :math:`[0, 1[`.
    snr : float or np.inf
        Signal-to-noise ratio.
    w_true : np.array, shape (n_features,) or (n_features, n_tasks)| None
        True regression coefficients. If None, a sparse array with standard
        Gaussian non zero entries is simulated.
    density : float
        Proportion of non zero elements in w_true if the latter is simulated.
    X_density : float in ]0, 1]
        Proportion of elements of X which are non-zero.
    random_state : int | RandomState instance | None (default)
        Determines random number generation for data generation. Use an int to
        make the randomness deterministic.
    Returns
    -------
    X : ndarray or CSC matrix, shape (n_samples, n_features)
        A design matrix with Toeplitz covariance.
    y : ndarray, shape (n_samples,) or (n_samples, n_tasks)
        Observation vector/matrix.
    w_true : ndarray, shape (n_features,) or (n_features, n_tasks)
        True regression vector/matrix of the model.
    """
    if not 0 <= rho < 1:
        raise ValueError("The correlation `rho` should be chosen in [0, 1[.")
    if not 0 < density <= 1:
        raise ValueError("The density should be chosen in ]0, 1].")
    if not 0 < X_density <= 1:
        raise ValueError("The density of X should be chosen in ]0, 1].")
    if snr < 0:
        raise ValueError("The snr should be chosen in [0, inf].")
    rng = check_random_state(random_state)
    nnz = int(density * n_features)

    if rho != 0:
        # X is generated cleverly using an AR model with reason corr and i
        # innovation sigma^2 = 1 - \rho ** 2: X[:, j+1] = rho X[:, j] + eps_j
        # where eps_j = sigma * rng.randn(n_samples)
        sigma = np.sqrt(1 - rho * rho)
        U = rng.randn(n_samples)

        X = np.empty([n_samples, n_features], order="F")
        X[:, 0] = U
        for j in range(1, n_features):
            U *= rho
            U += sigma * rng.randn(n_samples)
            X[:, j] = U
    else:
        X = rng.randn(n_samples, n_features)

    if X_density != 1:
        zeros = rng.binomial(n=1, size=X.shape, p=1 - X_density).astype(bool)
        X[zeros] = 0.0
        from scipy import sparse

        X = sparse.csc_matrix(X)

    if w_true is None:
        w_true = np.zeros((n_features, n_tasks))
        support = rng.choice(n_features, nnz, replace=False)
        w_true[support, :] = rng.randn(nnz, n_tasks)
    else:
        if w_true.ndim == 1:
            w_true = w_true[:, None]

    Y = X @ w_true
    noise = rng.randn(n_samples, n_tasks)
    if snr not in [0, np.inf]:
        Y += noise / norm(noise) * norm(Y) / snr
    elif snr == 0:
        Y = noise

    if n_tasks == 1:
        return X, Y.flatten(), w_true.flatten()
    else:
        return X, Y, w_true


def make_correlated_design(n_samples, n_features, rho, random_state):

    rng = check_random_state(random_state)

    if rho != 0:
        sigma = np.sqrt(1 - rho * rho)
        U = rng.randn(n_samples)

        X = np.empty([n_samples, n_features], order="F")
        X[:, 0] = U
        for j in range(1, n_features):
            U *= rho
            U += sigma * rng.randn(n_samples)
            X[:, j] = U
    else:
        X = rng.randn(n_samples, n_features)

    return X


def make_jordan_se1(
    n_samples=200, n_features=10, rho=0.6, noise_level=0.1, random_state=None
):
    rng = check_random_state(random_state)

    X = make_correlated_design(n_samples, n_features, rho, random_state)

    y_noise = X[:, 0] + noise_level * rng.randn(n_samples)
    y = y_noise.reshape(-1, 1)
    return X, y


def make_jordan_se2(
    n_samples=300, n_features=10, rho=0.5, noise_level=0.1, random_state=None
):
    rng = check_random_state(random_state)

    X = make_correlated_design(n_samples, n_features, rho, random_state)

    y_noise = X[:, 0] ** 3 + X[:, 1] ** 3 + noise_level * rng.randn(n_samples)
    y = y_noise.reshape(-1, 1)
    return X, y


def make_jordan_se3(n_samples=200, n_features=10, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, n_features)

    y_noise = X[:, 0] * X[:, 1] + noise_level * rng.randn(n_samples)
    y = y_noise.reshape(-1, 1)
    return X, y


def make_gregorova_se1(
    n_samples=2000, noise_level=0.1, random_state=None
):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, 18)

    y_noise = np.sin(np.square(X[:, 0] + X[:, 2])) * np.sin(
        X[:, 6] * X[:, 7] * X[:, 8]
    ) + noise_level * rng.randn(n_samples)
    y = y_noise.reshape(-1, 1)
    return X, y


def make_gregorova_se2(n_samples=2000, noise_level=0.1, random_state=None):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, 100)

    y_noise = np.log(np.square(np.sum(X[:, 10:15], axis=1))) + noise_level * rng.randn(
        n_samples
    )
    y = y_noise.reshape(-1, 1)
    return X, y


def make_vehtari(n_samples=300, noise_level=0.3, random_state=None):
    rng = check_random_state(random_state)

    X = -1 + 2 * rng.rand(n_samples, 8)

    phi = np.linspace(np.pi / 10, np.pi, 8)
    coefs = np.sqrt(2.0 / (1 - np.sin(phi) * np.cos(phi) / phi))

    y_noise = np.sin(X * phi) @ coefs + noise_level * rng.randn(n_samples)

    y = y_noise.reshape(-1, 1)
    return X, y
