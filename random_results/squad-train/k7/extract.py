import os
import time


def extract_last_four_lines(directory):
    with open(os.path.join(directory, "summary.txt"), "a") as summary_file:
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                with open(os.path.join(directory, filename), "r") as file:
                    lines = file.readlines()
                    last_four_lines = lines[-4:]
                    summary_file.write(f"Last four lines of {filename}:\n")
                    for line in last_four_lines:
                        summary_file.write(line)
                    summary_file.write("\n")


def grep(file, string):
    to_extract = ""
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if string in line:
                # print(line, end="")
                for j in range(1, 4):
                    if i + j < len(lines):
                        # print(lines[i + j], end="")
                        to_extract += lines[i + j]

    # print(to_extract)
    splitted = to_extract.split("\n")
    total = 0
    count = 0
    for line in splitted:
        if "Average Semantic Similarity:" in line:
            # print(line)
            number = line[line.find(":") + 2 :]
            # print(number)
            total += float(number)
            count += 1

    print(string, end="\t")
    result = total / count
    print(f"{result:.5f}")


# Example usage:
# extract_last_four_lines('/path/to/directory')

pending = [
    "bm25_top1.",
    "bm25_top3.",
    "bm25_top5.",
    "bm25_top10.",
    "bm25_top20.",
    "openai_top1.",
    "openai_top3.",
    "openai_top5.",
    "openai_top10.",
    "openai_top20.",
    "kvcache.",
    "kvcache_nokv.",
]

if __name__ == "__main__":
    extract_last_four_lines(".")

    time.sleep(3)

    # grep("./summary.txt", "bm25_top1.")
    for p in pending:
        grep("./summary.txt", p)

