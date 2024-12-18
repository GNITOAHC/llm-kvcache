import os
import time
import re


def extract_last_four_lines(directory):
    with open(os.path.join(directory, "summary_rt.txt"), "a") as summary_file:
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
        if "retrieve time:" in line:
            # print(line)
            retrieve_time_match = re.search(r"retrieve time: ([0-9.]+)", line)
            retrieve_time = float(retrieve_time_match.group(1)) if retrieve_time_match else None
            # print(retrieve_time)
            # number = line[line.find(":") + 2 :]
            # print(number)
            if not retrieve_time:
                print("Error in parsing retrieve time")
                continue
            total += float(retrieve_time)
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
]

if __name__ == "__main__":
    extract_last_four_lines("./k7/")

    time.sleep(3)

    # grep("./summary.txt", "bm25_top1.")
    for p in pending:
        grep("./k7/summary_rt.txt", p)
