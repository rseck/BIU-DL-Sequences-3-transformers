import os
import math

# Input and output file paths
input_file = 'data/input.txt'
output_file_95 = 'data/train/train.txt'
output_file_5 = 'data/dev/dev.txt'

# Read all lines from the input file
with open(input_file, 'r') as f:
    lines = f.readlines()

# Calculate the split point
total_lines = len(lines)
split_point = max(0, total_lines - 500)

# Write the first lines to one file
with open(output_file_95, 'w') as f:
    f.writelines(lines[:split_point])

# Write the last  1280 lines to another file
with open(output_file_5, 'w') as f:
    f.writelines(lines[split_point:])

print(f"Total lines: {total_lines}")
print(f"Lines in 95% file: {split_point}")
print(f"Lines in 5% file: {total_lines - split_point}")