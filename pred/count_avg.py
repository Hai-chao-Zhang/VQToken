import re
import sys

def extract_cluster_indices(filename):
    pattern = re.compile(r"################ ######## #### ## # Cluster indices shape:\s*(\d+)")
    values = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                values.append(int(match.group(1)))
    
    if values:
        average = sum(values) / len(values)
        print(f"Average Cluster Indices Shape: {average:.2f}")
    else:
        print("No matching lines found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
    else:
        extract_cluster_indices(sys.argv[1])