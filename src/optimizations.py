import sys
import itertools
import time
import pulp
import numpy as np
from sklearn.cluster import KMeans
from cvxopt import matrix
from cvxopt import solvers


def get_soft_const_matrix(const, N, K):
    C = []
    M = len(const)
    C = np.zeros((N*K, N*K))
    for m, cat_const in enumerate(const):
        n1, n2 = cat_const
        if n1 > n2:
            buff = n1
            n1 = n2
            n2 = buff
        for k in range(K):
            C[n1*K+k, n1*K+k] += 1
            C[n2*K+k, n1*K+k] += -1
            C[n1*K+k, n2*K+k] += -1
            C[n2*K+k, n2*K+k] += 1

    return C


def get_const_matrix(must, cannot, N, K):
    C = []
    d = []
    # make cannot-link matrix
    Mu = len(must)
    Ca = len(cannot)
    if Ca > 0:
        d.extend([1]*Ca*K)
        for m, cat_const in enumerate(cannot):
            n1, n2 = cat_const
            for k in range(K):
                c = [0]*(N*K + Mu*K)
                c[n1*K+k] = 1
                c[n2*K+k] = 1        
                C.append(c)
    # make must-link matrix
    if Mu > 0:
        d.extend([0]*Mu*K*2)
        for m, cat_const in enumerate(must):
            n1, n2 = cat_const
            for k in range(K):
                c1 = [0]*(N*K + Mu*K)
                c2 = [0]*(N*K + Mu*K)
                c1[n1*K+k] = -1
                c1[n2*K+k] = 1
                c2[n1*K+k] = 1
                c2[n2*K+k] = -1
                c1[N*K+m*K+k] = -1
                c2[N*K+m*K+k] = -1
                C.append(c1)
                C.append(c2)

    return np.array(C), np.array(d)


def get_pseudo_soft_labeling_with_mip(inputs):
    prob, const, r = inputs
    N, K = prob.shape
    M = len(const)
    P = [prob[n, k] + sys.float_info.epsilon if prob[n, k] == 0 else prob[n, k] for n, k in itertools.product(range(N), range(K))]
    P.extend([-r]*M*K)
    P[:N*K] = np.log(P[:N*K])
    print(P)

    problem = pulp.LpProblem("Problem", pulp.LpMaximize)

    x = []
    # 0-1変数を宣言
    for n, k in itertools.product(range(N), range(K)):
        x.append(pulp.LpVariable(f"x({n},{k})", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous))
    # 連続変数を宣言
    for m, k in itertools.product(range(M), range(K)):
        x.append(pulp.LpVariable(f"m({m},{k})", lowBound=-10**10, upBound=10**10, cat=pulp.LpContinuous))
    # pulp.LpContinuous : 連続変数
    # pulp.LpInteger    : 整数変数
    # pulp.LpBinary     : 0-1変数

    # 制約の行列を宣言
    C = get_const_matrix(const=const, N=N, K=K)
    # 目的関数を宣言
    problem += pulp.lpSum(P[i] * x[i] for i in range(len(P))), "TotalCost"

    # 制約を満たすかどうか
    for m in range(C.shape[0]):
        problem += sum(C[m, i] * x[i] for i in range(C.shape[1])) <= 0, f"Constraint_leq_{m}"

    # 各サンプルごとに選ばれるクラスは１つ
    for n in range(N):
        problem += sum(x[n*K+k] for k in range(K)) == 1, f"Constraint_eq_{n}"

    # ソルバー指定
    solver = pulp.PULP_CBC_CMD()
    # pulp.PULP_CBC_CMD() : PuLP付属のCoin-CBC
    # pulp.GUROBI_CMD()   : Gurobiをコマンドラインから起動 (.lpファイルを一時生成)
    # pulp.GUROBI()       : Gurobiをライブラリーから起動 (ライブラリーの場所指定が必要)
    # ほかにもいくつかのソルバーに対応
    # (使用例)
    # if pulp.GUROBI_CMD().available():
    #     solver = pulp.GUROBI_CMD()

    result_status = problem.solve(solver)
    for xx in x:
        print(xx.name, '=', xx.value())
    # X = []
    # for n, k in itertools.product(range(N), range(K)):
    #     X.append(x[n*K+k].value())
    X = np.array([[x[n*K+k].value() for k in range(K)] for n in range(N)])

    return X


def get_pseudo_soft_labeling_with_qp(inputs):
    prob, const, r = inputs
    N, K = prob.shape
    M = len(const)
    q = [prob[n, k] + sys.float_info.epsilon if prob[n, k] == 0 else prob[n, k] for n, k in itertools.product(range(N), range(K))]
    q = -1 * np.log(q)
    q = matrix(q.astype(np.double))

    # 制約の行列を宣言
    P = r * get_soft_const_matrix(const=const, N=N, K=K)
    P = matrix(P.astype(np.double))
    A = np.zeros((N, N*K))
    for n in range(N):
        A[n, n*K:(n+1)*K] = 1
    A = matrix(A) 
    b = matrix(np.ones(N))
    G = matrix(np.append(np.eye(N*K), -np.eye(N*K), axis=0))
    h = matrix(np.append(np.ones(N*K), np.zeros(N*K), axis=0))
    solvers.options['show_progress'] = False

    # 目的関数を宣言
    sol = solvers.qp(P=P, q=q, G=G, h=h, A=A, b=b)
    X = np.zeros((N, K))
    for n in range(N):
        X[n] = [sol['x'][n*K+k] for k in range(K)]

    return X


class hard_mip_with_link():
    def __init__(self, inputs):
        dist, const, self.r = inputs
        self.N, self.K = dist.shape
        self.M = len(const[0]) if len(const) == 2  else len(const)
        # 制約の行列を宣言
        self.C, self.d = get_const_matrix(must=const[0], cannot=const[1], N=self.N, K=self.K)
    
    def fit(self, dist):
        if dist.shape != (self.N, self.K):
            return
        Y = [dist[n, k] + sys.float_info.epsilon if dist[n, k] == 0 else dist[n, k] for n, k in itertools.product(range(self.N), range(self.K))]
        Y.extend([self.r]*self.M*self.K)
        problem = pulp.LpProblem("Problem", pulp.LpMinimize)
        x = []
        # 0-1変数を宣言
        for n, k in itertools.product(range(self.N), range(self.K)):
            x.append(pulp.LpVariable(f"x({n},{k})", 0, 1, pulp.LpBinary))
        # 連続変数を宣言
        for m, k in itertools.product(range(self.M), range(self.K)):
            x.append(pulp.LpVariable(f"m({m},{k})", -10**10, 10**10, pulp.LpContinuous))
        # pulp.LpContinuous : 連続変数
        # pulp.LpInteger    : 整数変数
        # pulp.LpBinary     : 0-1変数

        # 目的関数を宣言
        problem += pulp.lpSum(Y[i] * x[i] for i in range(len(Y))), "TotalCost"

        # 制約を満たすかどうか
        for m, d in zip(range(self.C.shape[0]), self.d):
            problem += sum(self.C[m, i] * x[i] for i in range(self.C.shape[1])) <= d, f"Constraint_leq_{m}"

        # 各サンプルごとに選ばれるクラスは１つ
        for n in range(self.N):
            problem += sum(x[n*self.K+k] for k in range(self.K)) == 1, f"Constraint_eq_{n}"

        # ソルバー指定
        solver = pulp.PULP_CBC_CMD()
        # pulp.PULP_CBC_CMD() : PuLP付属のCoin-CBC
        # pulp.GUROBI_CMD()   : Gurobiをコマンドラインから起動 (.lpファイルを一時生成)
        # pulp.GUROBI()       : Gurobiをライブラリーから起動 (ライブラリーの場所指定が必要)
        # ほかにもいくつかのソルバーに対応
        # (使用例)
        # if pulp.GUROBI_CMD().available():
        #     solver = pulp.GUROBI_CMD()

        result_status = problem.solve(solver)
        X = np.zeros((self.N, self.K))
        for n, k in itertools.product(range(self.N), range(self.K)):
            X[n, k] = x[n*self.K+k].value()

        return X.T

def soft_const_clustering(X, K, const, r):
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    d = np.array([np.linalg.norm(X-kmeans.cluster_centers_[k], axis=1) for k in range(K)]).T
    old_center = kmeans.cluster_centers_
    for _ in range(1000):
        opt = hard_mip_with_link(inputs=(d, const, r))
        C = opt.fit(d)
        cluidx = np.argmax(C, axis=0)
        center = np.array([np.mean(X[cluidx==k], axis=0) for k in range(K)])
        if np.linalg.norm(old_center - center) < 0.0001:
            print('OPTIMIZED !!')
            return cluidx, kmeans.labels_
        d = np.array([np.linalg.norm(X-center[k, None], axis=1) for k in range(K)]).T
        old_center = center
    
    return cluidx, kmeans.labels_

if __name__ == '__main__':
    import scipy.io
    Dictionary = scipy.io.loadmat('savefile.mat')

    n = 10
    p = []
    for m, s in zip([0, 1, 3], [0.3, 0.1, 0.5]):
        p.append(np.random.normal(m, s, (n, 2)))
    p = np.concatenate(p, axis=0)
    print(p.shape)
    const =  [[[0, 29]], 
              [[0, 19], [0, 2]]]
    labels, k = soft_const_clustering(p, 10, const, 50)

    print(k)
    print(labels)