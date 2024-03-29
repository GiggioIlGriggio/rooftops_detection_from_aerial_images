from scipy.interpolate import griddata
import numpy as np
from scipy.sparse import csr_matrix

shape = (14790, 12816)
num_nonzero = 488686

random_rows = np.random.randint(0, shape[0], num_nonzero)
random_cols = np.random.randint(0, shape[1], num_nonzero)

random_values = np.random.rand(num_nonzero).astype(np.float32)

sparse_matrix = csr_matrix((random_values, (random_rows, random_cols)), shape=shape, dtype=np.float32)
sparse_matrix = sparse_matrix.toarray()

coords = np.column_stack(np.nonzero(sparse_matrix))
values = sparse_matrix[coords[:, 0], coords[:, 1]]
grid_x, grid_y = np.mgrid[0:sparse_matrix.shape[0], 0:sparse_matrix.shape[1]]
depth_mask_cut = griddata(coords, values, (grid_x, grid_y), method='cubic')
