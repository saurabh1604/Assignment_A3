import numpy as np

# Dataset
X = np.array([1, 2, 4, 6, 8])
Y = np.array([3, 5, 9, 13, 17])
N = len(X)

def get_loss(w0, w1):
    y_pred = w0 + w1 * X
    error = y_pred - Y
    loss = 0.5 * np.sum(error**2)
    return loss

def get_gradient(w0, w1):
    y_pred = w0 + w1 * X
    error = y_pred - Y
    grad_w0 = np.sum(error)
    grad_w1 = np.sum(error * X)
    return np.array([grad_w0, grad_w1])

def get_hessian():
    h_00 = N
    h_01 = np.sum(X)
    h_10 = np.sum(X)
    h_11 = np.sum(X**2)
    return np.array([[h_00, h_01], [h_10, h_11]])

# Part (a)
print("--- Part (a) ---")
hessian = get_hessian()
print(f"Hessian:\n{hessian}")
eigenvalues = np.linalg.eigvals(hessian)
print(f"Eigenvalues of Hessian: {eigenvalues}")
print(f"Is Convex: {np.all(eigenvalues > 0)}")

# Part (b)
print("\n--- Part (b) ---")
# Gradient formula is printed in latex later, we just need the function.

# Part (c)
print("\n--- Part (c) ---")
w_start = np.array([2.0, 2.0])

# Scenario 1: Constant LR
print("Scenario 1 (Constant LR = 0.05):")
w = w_start.copy()
eta = 0.05
for i in range(1, 3):
    grad = get_gradient(w[0], w[1])
    print(f"Iteration {i-1} (Start): w = {w}, J = {get_loss(w[0], w[1])}, Grad = {grad}")
    w = w - eta * grad
    print(f"Iteration {i} (End): w = {w}, J = {get_loss(w[0], w[1])}")

# Scenario 2: Decaying LR
print("\nScenario 2 (Decaying LR):")
w = w_start.copy()
eta_0 = 0.1
k = 0.4
for i in range(1, 3): # t = 1, 2
    eta_t = eta_0 * np.exp(-k * i)
    grad = get_gradient(w[0], w[1])
    print(f"Iteration {i-1} (Start): w = {w}, J = {get_loss(w[0], w[1])}, Grad = {grad}, eta_{i} = {eta_t}")
    w = w - eta_t * grad
    print(f"Iteration {i} (End): w = {w}, J = {get_loss(w[0], w[1])}")


# Part (e) Line Search
print("\n--- Part (e) ---")
# d = -grad(w_start)
grad_start = get_gradient(w_start[0], w_start[1])
d = -grad_start
print(f"Search direction d = {d}")

def phi(eta):
    w_new = w_start + eta * d
    return get_loss(w_new[0], w_new[1])

# Binary Search
print("\nBinary Search:")
a, b = 0.0, 1.0
print(f"Initial Interval: [{a}, {b}]")
for i in range(1, 3):
    m = (a + b) / 2
    val_m = phi(m)
    val_m_eps = phi(m + 1e-3)

    # Derivative approximation: (phi(m+eps) - phi(m))/eps
    # If increasing (val_m < val_m_eps), min is to left.
    # If decreasing (val_m > val_m_eps), min is to right.

    print(f"Iter {i}: m = {m}, phi(m) = {val_m}, phi(m+eps) = {val_m_eps}")

    if val_m < val_m_eps:
        # Increasing, go left
        b = m
        print(f"  Result: Increasing. New Interval: [{a}, {b}]")
    else:
        # Decreasing, go right
        a = m
        print(f"  Result: Decreasing. New Interval: [{a}, {b}]")

# Golden Section Search
print("\nGolden Section Search (M1=1/4, M2=3/4):")
a, b = 0.0, 1.0
print(f"Initial Interval: [{a}, {b}]")
for i in range(1, 3):
    # Using relative 1/4 and 3/4
    width = b - a
    m1 = a + 0.25 * width
    m2 = a + 0.75 * width

    val_m1 = phi(m1)
    val_m2 = phi(m2)

    print(f"Iter {i}: M1 = {m1}, M2 = {m2}, phi(M1) = {val_m1}, phi(M2) = {val_m2}")

    if val_m1 < val_m2:
        # Min is in [a, m2]
        b = m2
        print(f"  Result: phi(M1) < phi(M2). New Interval: [{a}, {b}]")
    else:
        # Min is in [m1, b]
        a = m1
        print(f"  Result: phi(M1) >= phi(M2). New Interval: [{a}, {b}]")
