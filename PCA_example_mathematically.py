#PCA
import numpy as np

def normalize(data_set,n_features,n_samples):
    mean=np.zeros(n_features)
    for i in range(n_features):
        s=0
        for j in range(n_samples):
            s=s+data_set[j][i]
            if i==0:
                continue
            data_set[j][i-1]-=mean[i-1]
        mean[i]=s/n_samples
    for j in range(n_samples):
        data_set[j][n_features-1]-=mean[n_features-1]
    return data_set

def covariance_matrix(data_set,n_features,n_samples):
    covariance_matrix=np.zeros([n_features,n_features])
    for r in range(n_features):
        for c in range(n_features):
            s=0
            for i in range(n_samples):
                s+=(data_set[i][r]*data_set[i][c])
            s=s/n_samples
            covariance_matrix[r][c]=s
    return covariance_matrix

def calc_result(eigen_values):
    mean_eigen=sum(eigen_values)
    for i in range(len(eigen_values)):
        print("FEATURE ", i," : ",eigen_values[i]/mean_eigen)

def make_feature_vector(eigen_vector,size):
    return eigen_vector[:,0:size]
        
if __name__ == '__main__':
    data_set=np.array([[10,6,12,5],[11,4,9,20],[8,5,10,6],[3,3,2.5,2],[2,2.8,1.3,18],[1,1,2,19]])
    n_features=4
    n_samples=6
    print(data_set)
    data_set=normalize(data_set,n_features,n_samples)
    print(data_set)
    covariance_matrix=covariance_matrix(data_set,n_features,n_samples)
    print(covariance_matrix)
    eigen_values,eigen_vector=np.linalg.eig(covariance_matrix)
    print("eigen_values:",eigen_values)
    print("eigen_vectors:",eigen_vector)
    calc_result(eigen_values)
    size=2
    feature_vector=make_feature_vector(eigen_vector,size)
    print(feature_vector)
    
