import numpy as np 


def kmeans_clustering(k, passed_x):
    z = [[0]*passed_x.shape[0] for i in range(k)]
    # initializing cluster centers
    mus = passed_x[:k].copy()
    # mus = np.float32(mus)
    # print(mus)
    for j in range(k):
        z[j][j] = 1
    # print(z)
    # flag
    update = True
    while update:
        update = False
        for i in range(passed_x.shape[0]):
            norms = np.linalg.norm(mus - passed_x[i], axis=1)
            assigned_cluster = np.argmin(norms)
            # cluster assignment
            for j in range(k):
                if j != assigned_cluster:
                    z[j][i] = 0
                else:
                    if z[assigned_cluster][i] != 1:
                        update = True
                    z[assigned_cluster][i] = 1
        
        # updating the cluster centers
        for j in range(k):
            Nj = sum(z[j])
            vec_sum = np.array([0.0]*passed_x.shape[1])
            for i in range(passed_x.shape[0]):
                vec_sum += z[j][i]*passed_x[i]

            mus[j] = vec_sum/Nj

    return mus

x = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [1, 3, 5, 7, 9, 11]], dtype=np.float)
print(kmeans_clustering(2, x))