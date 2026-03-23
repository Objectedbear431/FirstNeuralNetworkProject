import numpy as np

# Input data: each row has two numbers
X = np.array([
    [1, 1], # each ROW is a single EXAMPLE. each example has 2 numbers....
    [1, 2],         # ^think of each row as a pattern the network should look at
    [2, 1],
    [2, 2],         
    [3, 2],     # shape of X:
    [2, 3],         # ^(12, 2) this means: 12 rows (EXAMPLES) and 2 columns (FEATURES)
    [8, 8],
    [8, 9],
    [9, 8],
    [9, 9],
    [7, 8],
    [8, 7]
], dtype=float)


# Labels:
# 0 = Class A
# 1 = Class B

y = np.array([
    [0], # each ROW here is an ANSWER to each EXAMPLE
    [0],    
    [0],    # shape of y: 
    [0],        # ^(12, 1) this means: 12 EXAMPLES and 1 OUTPUT PER EXAMPLE
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=float)

# The neural network is learning the pattern that smaller numbers belong to Class A and larger numbers belong to
# Class B. The matters because neural networks don't "know math" nor "understand numbers". It only sees PATTERNS    
# and their associated LABELS, and it tries to learn their relationship.
