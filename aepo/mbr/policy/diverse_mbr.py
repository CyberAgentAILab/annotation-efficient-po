import numpy as np


def generate_objective(k, div_pen, matrix):
    """Generate an objective function for the diverse MBR."""

    def objective(X: np.array):
        n = matrix.shape[0]
        matrix_ = np.copy(matrix)
        np.fill_diagonal(matrix, 0.0) # Diagonals are not used in the objective.
        # The normalization is applied to make div_pen be in a range of (0, 1). 
        # similarity has (n - 1) * k comparisons to give the score, so normalized so.
        # diversity has k * (k - 1) comparisons to give the score, so normalized so.
        
        # The larger the score is, the more similar it is to the distribution.
        similarity = np.ones((1, n)) @ matrix_ @ X / (n-1) / k
        # The smaller the score is, the larger the diversity is.
        diveristy = -np.transpose(X) @ matrix_ @ X * div_pen / k / (k-1)

        return similarity + diveristy
    
    return objective

def gbfs(func, n, k):
    """Greedy Best First Search for the diverse MBR."""
        
    node = np.zeros(n, dtype=int)

    for nsize in range(k):

        # TODO: This procedure can be refactored with lambda
        cur_best, cur_best_i = -np.inf, -1
        for i in range(n):
            if node[i] == 1:
                continue
            next_node = np.copy(node)
            next_node[i] = 1
            obj = func(next_node)

            if obj > cur_best:
                cur_best = obj
                cur_best_i = i
        node[cur_best_i] = 1

    return node

def local_search(func, init, iterations=100, neighbor=2):
    """Local search for the diverse MBR."""
    n = init.shape[0]
    k = sum(init)

    node = np.copy(init)

    for i in range(iterations):
        cur_node = np.copy(node)
        neighbor = neighbor
        indices = np.where(node == 1)[0]
        removed_cand = np.random.choice(k, neighbor, replace=False)

        for l in range(neighbor):
            node[indices[removed_cand[l]]] = 0

        for l in range(neighbor):
            cur_best, cur_best_i = -np.inf, -1
            for i in range(n):
                if node[i] == 1:
                    continue
                next_node = np.copy(node)
                next_node[i] = 1
                obj = func(next_node)

                if obj > cur_best:
                    cur_best = obj
                    cur_best_i = i
            node[cur_best_i] = 1

        if func(node) < func(cur_node):
            node = cur_node
        
    return node

def compute_dmbr(hyp=None, score_function=None, matrix=None, weights=None, src=None, k=1, div_pen=0.0):
    """Compute the diverse MBR."""
    assert (score_function is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, score_function, [src] * len(hyp))
    
    n = matrix.shape[0]
    obj = generate_objective(k=k, div_pen=div_pen, matrix=matrix)
    gbfs_result = gbfs(obj, n, k=k)
    local_result = local_search(obj, gbfs_result, iterations=20, neighbor=1)

    k_bests = np.where(local_result == 1)[0]
    return k_bests
