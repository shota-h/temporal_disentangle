# coding: UTF-8

# 線形/整数線形最適化問題を解くためにPuLPをインポート
import pulp
# 計算時間を計るのにtimeをインポート
import time
import sys
import numpy as np

def test():
    # 作業員の集合（便宜上、リストを用いる）
    I = ["Aさん", "Bさん", "Cさん"]

    print(f"作業員の集合 I = {I}")


    # タスクの集合（便宜上、リストを用いる）
    J = ["仕事イ", "仕事ロ", "仕事ハ", 'M1']

    print(f"タスクの集合 J = {J}")


    # 作業員 i を タスク j に割り当てたときのコストの集合（一時的なリスト）
    cc = [
        [ 1.0, 0.0, 0.0],
        [ 0.6, 0.4, 0.0],
        [0.5, 0.5, 0.0],
        [10, 10, 10]
        ]
    # cc = [0.6, 0.4, 0.5, 0.5, 0.4, 0.6]
    cc[:3, :3] = np.log(cc[:3, :3])

    # cc はリストであり、添え字が数値なので、
    # 辞書 c を定義し、例えばcc[0][0] は c["Aさん","仕事イ"] でアクセスできるようにする
    c = {} # 空の辞書
    for i in I:
        for j in J:
            c[i,j] = cc[I.index(i)][J.index(j)]
    sys.exit()
    print("コスト c[i,j]: ")
    for i in I:
        for j in J:
            print(f"c[{i},{j}] = {c[i,j]:2},  ", end = "")
        print("")
    print("")



    # 数理最適化問題（最小化）を宣言
    problem = pulp.LpProblem("Problem-2", pulp.LpMaximize)
    # pulp.LpMinimize : 最小化 
    # pulp.LpMaximize : 最大化


    # 変数集合を表す辞書
    x = {} # 空の辞書
        # x[i,j] または x[(i,j)] で、(i,j) というタプルをキーにしてバリューを読み書き

    # 0-1変数を宣言
    for i in I:
        for j in J:
            x[i,j] = pulp.LpVariable(f"x({i},{j})", 0, 1, pulp.LpBinary)
            # 変数ラベルに '[' や ']' や '-' を入れても、なぜか '_' に変わる…？
    # lowBound, upBound を指定しないと、それぞれ -無限大, +無限大 になる

    # 内包表記も使える
    # x_suffixes = [(i,j) for i in I for j in J]
    # x = pulp.LpVariable.dicts("x", x_suffixes, cat = pulp.LpBinary) 

    # pulp.LpContinuous : 連続変数
    # pulp.LpInteger    : 整数変数
    # pulp.LpBinary     : 0-1変数


    # 目的関数を宣言
    problem += pulp.lpSum(c[i,j] * x[i,j] for i in I for j in J), "TotalCost"
    # problem += sum(c[i,j] * x[i,j] for i in I for j in J)
    # としてもOK

    # 制約条件を宣言
    # 各作業員 i について、割り当ててよいタスク数は1つ以下
    for i in I:
        problem += sum(x[i,j] for j in J) <= 1, f"Constraint_leq_{i}"
        # 制約条件ラベルに '[' や ']' や '-' を入れても、なぜか '_' に変わる…？

    # 各タスク j について、割り当てられる作業員数はちょうど1人
    for j in J:
        problem += sum(x[i,j] for i in I) == 1, f"Constraint_eq_{j}"


    # 問題の式全部を表示
    print("問題の式")
    print(f"-" * 8)
    print(problem)
    print(f"-" * 8)
    print("")



    # 計算
    # ソルバー指定
    solver = pulp.PULP_CBC_CMD()
    # pulp.PULP_CBC_CMD() : PuLP付属のCoin-CBC
    # pulp.GUROBI_CMD()   : Gurobiをコマンドラインから起動 (.lpファイルを一時生成)
    # pulp.GUROBI()       : Gurobiをライブラリーから起動 (ライブラリーの場所指定が必要)
    # ほかにもいくつかのソルバーに対応
    # (使用例)
    # if pulp.GUROBI_CMD().available():
    #     solver = pulp.GUROBI_CMD()

    # 時間計測開始
    time_start = time.perf_counter()

    result_status = problem.solve(solver)
    # solve()の()内でソルバーを指定できる
    # 何も指定しない場合は pulp.PULP_CBC_CMD()

    # 時間計測終了
    time_stop = time.perf_counter()



    # （解が得られていれば）目的関数値や解を表示
    print("計算結果")
    print(f"*" * 8)
    print(f"最適性 = {pulp.LpStatus[result_status]}, ", end="")
    print(f"目的関数値 = {pulp.value(problem.objective)}, ", end="")
    print(f"計算時間 = {time_stop - time_start:.3f} (秒)")
    print("解 x[i,j]: ")
    for i in I:
        for j in J:
            print(f"{x[i,j].name} = {x[i,j].value()},  ", end="")
        print("")
    print(f"*" * 8)


def for_myself():
    # 作業員の集合（便宜上、リストを用いる）
    N = 3
    K = 2
    M = 2
    r = -.2
    p11 = 1.0
    p21 = 0.3
    p31 = 1.0
    cc = [p11, 1-p11, p21, 1-p21, p31, 1-p31, r, r, r, r]
    cc[:N*K] = [cc[i] + sys.float_info.epsilon for i in range(N*K)]
    print(cc)
    cc[:N*K] = np.log(cc[:N*K])
    print(cc)

    # 数理最適化問題（最小化）を宣言
    problem = pulp.LpProblem("Problem-2", pulp.LpMaximize)
    # pulp.LpMinimize : 最小化 
    # pulp.LpMaximize : 最大化


    # 変数集合を表す辞書
    x = []

    # 0-1変数を宣言
    for n in range(N):
        for k in range(K):
            x.append(pulp.LpVariable(f"x({n},{k})", 0, 1, pulp.LpBinary))
    for m in range(M):
        for k in range(K):
            x.append(pulp.LpVariable(f"m({m},{k})", -10**10, 10**10, pulp.LpContinuous))
    # x[-1] = pulp.LpVariable(f"m({1},{0})", -10^10, 10^10, pulp.LpContinuous)
    # lowBound, upBound を指定しないと、それぞれ -無限大, +無限大 になる

    # 内包表記も使える
    # x_suffixes = [(i,j) for i in I for j in J]
    # x = pulp.LpVariable.dicts("x", x_suffixes, cat = pulp.LpBinary) 

    # pulp.LpContinuous : 連続変数
    # pulp.LpInteger    : 整数変数
    # pulp.LpBinary     : 0-1変数


    # 目的関数を宣言
    problem += pulp.lpSum(cc[i] * x[i] for i in range(len(cc))), "TotalCost"
    # problem += sum(c[i,j] * x[i,j] for i in I for j in J)
    # としてもOK


    # 制約条件を宣言
    # 各作業員 i について、割り当ててよいタスク数は1つ以下
    # for i in I:
    #     problem += sum(x[i] for j in J) <= 1, f"Constraint_leq_{i}"
        # 制約条件ラベルに '[' や ']' や '-' を入れても、なぜか '_' に変わる…？
    A = [
        [-1, 0, 1, 0, 0, 0, -1, 0, 0, 0],
        [0, -1, 0, 1, 0, 0, 0, -1, 0, 0],
        [0, 0, -1, 0, 1, 0, 0, 0, -1, 0],
        [0, 0, 0, -1, 0, 1, 0, 0, 0, -1],
        [1, 0, -1, 0, 0, 0, -1, 0, 0, 0],
        [0, 1, 0, -1, 0, 0, 0, -1, 0, 0],
        [0, 0, 1, 0, -1, 0, 0, 0, -1, 0],
        [0, 0, 0, 1, 0, -1, 0, 0, 0, -1],
        ]
    A = np.array(A)
    # 各タスク j について、割り当てられる作業員数はちょうど1人
    for m in range(A.shape[0]):
        problem += sum(A[m, i] * x[i] for i in range(A.shape[1])) <= 0, f"Constraint_leq_{m}"

    for n in range(N):
        problem += sum(x[n*K+k] for k in range(K)) == 1, f"Constraint_eq_{n}"


    # 問題の式全部を表示
    print("問題の式")
    print(f"-" * 8)
    print(problem)
    print(f"-" * 8)
    print("")



    # 計算
    # ソルバー指定
    solver = pulp.PULP_CBC_CMD()
    # pulp.PULP_CBC_CMD() : PuLP付属のCoin-CBC
    # pulp.GUROBI_CMD()   : Gurobiをコマンドラインから起動 (.lpファイルを一時生成)
    # pulp.GUROBI()       : Gurobiをライブラリーから起動 (ライブラリーの場所指定が必要)
    # ほかにもいくつかのソルバーに対応
    # (使用例)
    # if pulp.GUROBI_CMD().available():
    #     solver = pulp.GUROBI_CMD()

    # 時間計測開始
    time_start = time.perf_counter()

    result_status = problem.solve(solver)
    # solve()の()内でソルバーを指定できる
    # 何も指定しない場合は pulp.PULP_CBC_CMD()

    # 時間計測終了
    time_stop = time.perf_counter()



    # （解が得られていれば）目的関数値や解を表示
    print("計算結果")
    print(f"*" * 8)
    print(f"最適性 = {pulp.LpStatus[result_status]}, ", end="")
    print(f"目的関数値 = {pulp.value(problem.objective)}, ", end="")
    print(f"計算時間 = {time_stop - time_start:.3f} (秒)")
    print("解 x[i,j]: ")
    for i in range(len(x)):
        print(f"{x[i].name} = {x[i].value()},  ", end="\n")
    # for i in I:
    #     for j in J:
    #         print(f"{x[i,j].name} = {x[i,j].value()},  ", end="")
    #     print("")
    # print(f"*" * 8)

if __name__ == '__main__':
    for_myself()