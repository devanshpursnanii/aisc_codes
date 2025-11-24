import numpy as np

def max_min_composition(R1, R2):
    """
    Computes fuzzy composition R = R1 ∘ R2 using max-min method.
    """
    v_count, r_count = R1.shape
    r_count2, d_count = R2.shape

    if r_count != r_count2:
        raise ValueError("Dimension mismatch: inner dimensions of R1 and R2 must match.")

    R = np.zeros((v_count, d_count))

    for i in range(v_count):         # iterate over V
        for j in range(d_count):     # iterate over D
            mins = []
            for k in range(r_count): # iterate over R
                mins.append(min(R1[i][k], R2[k][j]))
            R[i][j] = max(mins)
    return R


def max_product_composition(R1, R2):
    """
    Computes fuzzy composition R = R1 ∘ R2 using max-product method.
    """
    v_count, r_count = R1.shape
    r_count2, d_count = R2.shape

    if r_count != r_count2:
        raise ValueError("Dimension mismatch: inner dimensions must match.")

    R = np.zeros((v_count, d_count))

    for i in range(v_count):
        for j in range(d_count):
            prods = []
            for k in range(r_count):
                prods.append(R1[i][k] * R2[k][j])
            R[i][j] = max(prods)
    return R


# ---------------------------
# Example fuzzy relations
# ---------------------------

R1 = np.array([
    [0.8, 0.6, 0.4],
    [0.7, 0.9, 0.5],
    [0.5, 0.7, 0.8]
])

R2 = np.array([
    [0.9, 0.6, 0.3],
    [0.5, 0.8, 0.7],
    [0.4, 0.6, 0.9]
])

# Compute compositions
R_max_min = max_min_composition(R1, R2)
R_max_prod = max_product_composition(R1, R2)

print("Max-Min Composition:\n", R_max_min)
print("\nMax-Product Composition:\n", R_max_prod)
