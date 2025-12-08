import ast
from collections import defaultdict

def process_log_data(file_path):
    """
    Reads a log file, pairs entries by (width, depth), subtracts their
    values, and prints the results.

    The function assumes that for each (width, depth) combination, there are
    exactly two entries in the file. It subtracts the values of the second
    entry from the first.

    Args:
        file_path (str): The path to the input log file.
    """
    # Use a defaultdict to store lists of entries for each (width, depth) pair.
    # This simplifies adding new entries.
    paired_data = defaultdict(list)

    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Safely evaluate the string representation of the dictionary.
                # ast.literal_eval is much safer than eval().
                try:
                    record = ast.literal_eval(line.strip())
                    key = (record['width'], record['depth'])
                    paired_data[key].append(record)
                except (ValueError, SyntaxError) as e:
                    print(f"Warning: Could not parse line, skipping. Error: {e}\nLine: {line.strip()}")
                    continue

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # A list to hold the final calculated results.
    results = []

    # Iterate through the dictionary of paired data.
    # The items are sorted by width, then by depth to maintain a consistent order.
    for key, entries in sorted(paired_data.items()):
        # Ensure there are exactly two entries to form a pair.
        if len(entries) == 2:
            entry1, entry2 = entries[0], entries[1]

            # Calculate the difference for the specified fields.
            # The problem asks to subtract the second from the first.
            diff_magnetization = entry1['magnetization'] - entry2['magnetization']
            diff_square_magnetization = entry1['square_magnetization'] - entry2['square_magnetization']
            diff_seconds = entry1['seconds'] - entry2['seconds']

            # Create the result dictionary in the same format.
            # 'trial' is kept as 1, as in the original file.
            result = {
                'width': key[0],
                'depth': key[1],
                'trial': 1,
                'magnetization': diff_magnetization,
                'square_magnetization': diff_square_magnetization,
                'seconds': diff_seconds
            }
            results.append(result)
        else:
            # Handle cases where there isn't a pair of entries.
            print(f"Warning: Expected 2 entries for {key}, but found {len(entries)}. Skipping.")

    # Print each result dictionary in the desired format.
    for res in results:
        print(res)

# --- Execution ---
# To run this code, replace 'fullog-same.txt' with the actual path to your file
# if it's in a different directory.
if __name__ == "__main__":
    log_file = 'fullog.txt'
    process_log_data(log_file)

