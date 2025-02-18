r"""
    \multirow{6}{*}{Small}
        & Sparse RAG, Top-3   & 0.0008 & 0.7406 \\
        & Sparse RAG, Top-10  & 0.0012 & 1.5595 \\
        & Dense RAG, Top-3  & 0.4849 & 1.0093 \\
        & Dense RAG, Top-10 & 0.3803 & 2.6608 \\
        & CAG          & -      & 0.8512 \\
        & In-Context Learning      & -      & 9.3197 \\
    \midrule
    \multirow{6}{*}{Medium}
        & Sparse RAG, Top-3   & 0.0008 & 0.7148  \\
        & Sparse RAG, Top-10  & 0.0012 & 1.5306  \\
        & Dense RAG, Top-3  & 0.4140 & 0.9566 \\
        & Dense RAG, Top-10 & 0.4171 & 2.6361 \\
        & CAG          & -      & 1.4078 \\
        & In-Context Learning      & -      & 26.3717 \\
    \midrule 
    \multirow{6}{*}{Large}
        & Sparse RAG, Top-3   & 0.0008 & 0.6667 \\
        & Sparse RAG, Top-10  & 0.0012 & 1.5175 \\
        & Dense RAG, Top-3  & 0.4123 & 0.9331 \\
        & Dense RAG, Top-10 & 0.4100 & 2.6447 \\
        & CAG          & -      & 2.2631 \\
        & In-Context Learning      & -      & 92.0824 \\
    \bottomrule
"""

sm_bm25_top3 = [0.0008, 0.7148]
sm_bm25_top10 = [0.0012, 1.5595]
sm_dense_top3 = [0.4849, 1.0093]
sm_dense_top10 = [0.3803, 2.6608]
sm_cag = [0.8512, 0.0000]

md_bm25_top3 = [0.0008, 0.7148]
md_bm25_top10 = [0.0012, 1.5306]
md_dense_top3 = [0.4140, 0.9566]
md_dense_top10 = [0.4171, 2.6361]
md_cag = [1.4078, 0.0000]

lg_bm25_top3 = [0.0008, 0.6667]
lg_bm25_top10 = [0.0012, 1.5175]
lg_dense_top3 = [0.4123, 0.9331]
lg_dense_top10 = [0.4100, 2.6447]
lg_cag = [2.2631, 0.0005]

sm = [sm_bm25_top3, sm_bm25_top10, sm_dense_top3, sm_dense_top10, sm_cag]
md = [md_bm25_top3, md_bm25_top10, md_dense_top3, md_dense_top10, md_cag]
lg = [lg_bm25_top3, lg_bm25_top10, lg_dense_top3, lg_dense_top10, lg_cag]
data = [sm, md, lg]


threshold = 0.01
# threshold = 0.0

def main():
    largest_sum = max([sum(x) for x in sm + md + lg])

    print(f"Largest sum: {largest_sum}")
    normalized = lambda x: x / largest_sum

    for _, dataset in enumerate(data):
        for j, row in enumerate(dataset):
            for k, value in enumerate(row):
                if value < threshold:
                    dataset[j][k] = 0

    for _, dataset in enumerate(data):
        for j, row in enumerate(dataset):
            for k, value in enumerate(row):
                dataset[j][k] = normalized(value)

    for i, dataset in enumerate(data):
        print(f"Dataset {i + 1}")
        for row in dataset:
            print(f"& {row[0]:.3f} & {row[1]:.3f} \\\\")


if __name__ == "__main__":
    main()
