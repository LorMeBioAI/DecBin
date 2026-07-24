from tabulate import tabulate
import pandas as pd


def clusters_table(clusters, truth, tab, result_file):
    clx = set(clusters)
    trx = set(truth)
    c_map = {k: v for v, k in enumerate(clx)}
    t_map = {k: v for v, k in enumerate(trx)}

    matrix = [["_"] + [f"Bin-{c_map[x]}_({x})" for x in list(clx)]]
    matrix += [[x] + [0 for _ in range(len(clx))] for x in trx]

    mat = [[0 for _ in range(len(clx))] for _ in trx]

    for c, t in zip(clusters, truth):
        matrix[t_map[t] + 1][c_map[c] + 1] += 1
        mat[t_map[t]][c_map[c]] += 1

    matT = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]

    total_sum_mat = sum(sum(row) for row in mat)
    row_max_sum = sum(max(row) for row in mat)
    recall = row_max_sum / total_sum_mat

    total_sum_matT = sum(sum(row) for row in matT)
    col_max_sum = sum(max(row) for row in matT)
    precision = col_max_sum / total_sum_matT

    if tab:
        print(tabulate(matrix, tablefmt="plain"))

    dataframe = pd.DataFrame(matrix)
    dataframe.to_excel(result_file, index=False)

    f1 = 2 * recall * precision / (recall + precision)
    print(f"Precision\t{precision * 100:10.2f}")
    print(f"Recall    \t{recall * 100:10.2f}")
    print(f"F1-Score  \t{f1 * 100:10.2f}")
    print(f"Bins      \t{len(clx):10}")