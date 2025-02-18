import pandas as pd
import os
import re

def extract_top_k(method):
    """Extracts Top-k value from the method name."""
    match = re.search(r'_(\d+)$', method)
    return int(match.group(1)) if match else None

def read_data(file_path, dataset_name):
    """Reads a CSV file, renames necessary columns, and returns a DataFrame."""
    df = pd.read_csv(file_path)
    print(f"Columns in {file_path}: {df.columns.tolist()}\nRow count: {len(df)}")  # Debugging
    
    # Rename columns to match expected names
    df = df.rename(columns={
        "method": "System",
        "bert_score": dataset_name  # Assigning dataset name as column name (HotPotQA or SQuAD)
    })
    
    # Ensure scores are numeric
    df[dataset_name] = pd.to_numeric(df[dataset_name], errors='coerce')
    
    # Extract 'Top-k' values from method names
    df["Top-k"] = df["System"].apply(extract_top_k)
    df["System"] = df["System"].apply(lambda x: "Sparse RAG" if "bm25" in x else ("Dense RAG" if "dense" in x or "openapi" in x or "rag_openai" in x else ("CAG (Ours)" if "kvcache_withCAG" in x else x)))
    
    return df

def format_latex_table(small_data, medium_data, large_data):
    """Formats the data into a LaTeX table."""
    latex_str = r"""\
\begin{table}
  \caption{Experiment Results-1}
  \label{tab:results}
  \begin{tabular}{llccc}
    \toprule
    & & & HotPotQA & SQuAD \\
    Size & System & Top-$k$ & BERT-Score & BERT-Score \\
    \midrule
"""
    
    for size, data in zip(["Small", "Medium", "Large"], [small_data, medium_data, large_data]):
        latex_str += f"    \\multirow{{11}}{{*}}{{{size}}}\n"
        
        for system in ["Sparse RAG", "Dense RAG"]:
            subset = data[data['System'] == system]
            if not subset.empty:
                latex_str += f"      & \\multirow{{4}}{{*}}{{{system}}}\\\n"
                for k in [1, 3, 5, 10]:
                    row = subset[subset['Top-k'] == k]
                    if not row.empty:
                        hotpotqa_score = row.iloc[0]['HotPotQA']
                        squad_score = row.iloc[0]['SQuAD']
                        latex_str += f"      & & {k} & {hotpotqa_score:.4f} & {squad_score:.4f} \\\\ \n"
                latex_str += "      \\cline{2-5}\n"
        
        # CAG (Ours) row should come after Sparse and Dense RAG
        cag_row = data[data['System'] == "CAG (Ours)"]
        if not cag_row.empty:
            hotpotqa_score = cag_row.iloc[0]['HotPotQA']
            squad_score = cag_row.iloc[0]['SQuAD']
            latex_str += f"    & CAG (Ours) & & \\textbf{{{hotpotqa_score:.4f}}} & \\textbf{{{squad_score:.4f}}} \\\\ \n"
        
        latex_str += "    \midrule\n"
    
    latex_str += r"""    \bottomrule
  \end{tabular}
\end{table}
"""
    
    return latex_str

# Load data
small_squad = read_data("squad-train_small_summary.csv", "SQuAD")
small_hotpotqa = read_data("hotpotqa-train_small_summary.csv", "HotPotQA")
medium_squad = read_data("squad-train_medium_summary.csv", "SQuAD")
medium_hotpotqa = read_data("hotpotqa-train_medium_summary.csv", "HotPotQA")
large_squad = read_data("squad-train_large_summary.csv", "SQuAD")
large_hotpotqa = read_data("hotpotqa-train_large_summary.csv", "HotPotQA")

# Merge datasets
small_data = small_squad.merge(small_hotpotqa, on=["System", "Top-k"], how="inner")
medium_data = medium_squad.merge(medium_hotpotqa, on=["System", "Top-k"], how="inner")
large_data = large_squad.merge(large_hotpotqa, on=["System", "Top-k"], how="inner")

# Generate LaTeX table
latex_table = format_latex_table(small_data, medium_data, large_data)

# Save to file
with open("experiment_results.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table saved as 'experiment_results.tex'")
