import numpy as np

mat_x = np.array([[5,3,1],[9,6,3],[13,12,11]])
mat_y = np.array([[4,7,8],[22,45,76],[32,24,54]])

mat_xy = mat_x@mat_y
mat_xy_t = mat_xy.T
mat_xy_t2 = mat_xy.transpose()
mat_xy_inv = np.linalg.inv(mat_xy_t)

print(mat_x,'\n')
print(mat_y,'\n')
print(mat_xy.shape,'\n')
print(mat_xy_t,'\n')
print(mat_xy_t2,'\n')
print(mat_xy_inv.shape,'\n')