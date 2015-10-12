

import numpy as np
X = np.array([[1,1,2,3], [2, 1,4,5], [3, 2,4,5], [4, 1,2,1], [5, 4,3,1], [6, 1,4,3]])
from sklearn.decomposition import ProjectedGradientNMF
model = ProjectedGradientNMF(n_components=2, init='random', random_state=0)

print model.fit(X)
#ProjectedGradientNMF(beta=1, eta=0.1, init='random', max_iter=200,
#        n_components=2, nls_max_iter=2000, random_state=0, sparseness=None,
#        tol=0.0001)
print model.components_
#array([[ 0.77032744,  0.11118662],
#       [ 0.38526873,  0.38228063]])
print model.reconstruction_err_
#0.00746...

W = model.fit_transform(X);
H = model.components_;

print 'w: ' + str(W)
print 'h: ' + str(H)

model = ProjectedGradientNMF(n_components=2, sparseness='components', init='random', random_state=0)


print model.fit(X)
#ProjectedGradientNMF(beta=1, eta=0.1, init='random', max_iter=200,
#            n_components=2, nls_max_iter=2000, random_state=0,
#            sparseness='components', tol=0.0001)

print model.components_
#array([[ 1.67481991,  0.29614922],
#       [ 0.        ,  0.4681982 ]])
print model.reconstruction_err_