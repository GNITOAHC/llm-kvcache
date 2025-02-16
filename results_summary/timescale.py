r"""
\multirow{6}{*}{Small}
        & BM25 top-3   & 0.0009 & 0.6628 & -      & -      \\
        & BM25 top-10  & 0.0012 & 1.5139 & -      & -      \\
        & Dense top-3  & 0.4516 & 0.9280 & -      & -      \\
        & Dense top-10 & 0.4946 & 0.9280 & -      & -      \\
        & CAG          & -      & 0.7705 & 0.0005 & -      \\
        & w/o CAG      & -      & 8.6273 & -      & 0.0692 \\
    \midrule
    \multirow{6}{*}{Medium}
        & BM25 top-3   & 0.0008 & 0.6528 & -      & -      \\
        & BM25 top-10  & 0.0012 & 1.5202 & -      & -      \\
        & Dense top-3  & 0.4927 & 0.9138 & -      & -      \\
        & Dense top-10 & 0.4824 & 2.6128 & -      & -      \\
        & CAG          & -      & 1.2688 & 0.0005 & -      \\
        & w/o CAG      & -      & 25.5114 & -      & 0.3589 \\
    \midrule
    \multirow{6}{*}{Large}
        & BM25 top-3   & 0.0008 & 0.6667 & -      & -      \\
        & BM25 top-10  & 0.0012 & 1.5175 & -      & -      \\
        & Dense top-3  & 0.4123 & 0.9331 & -      & -      \\
        & Dense top-10 & 0.4100 & 2.6447 & -      & -      \\
        & CAG          & -      & 2.2631 & 0.0005 & -      \\
        & w/o CAG      & -      & 92.0824 & -      & 86.6877 \\
    \bottomrule
"""

sm_bm25_top3 = [0.0009, 0.6628]
sm_bm25_top10 = [0.0012, 1.5139]
sm_dense_top3 = [0.4516, 0.9280]
sm_dense_top10 = [0.4946, 0.9280]
sm_cag = [0.7705, 0.0005]

md_bm25_top3 = [0.0008, 0.6528]
md_bm25_top10 = [0.0012, 1.5202]
md_dense_top3 = [0.4927, 0.9138]
md_dense_top10 = [0.4824, 2.6128]
md_cag = [1.2688, 0.0005]

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
