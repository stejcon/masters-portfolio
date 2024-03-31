import json
import os
from collections import defaultdict


# Function to process JSON file and write refined data into a new JSON file
def process_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)

    # Dictionary to store statistics for each exit number
    exit_stats = defaultdict(
        lambda: {
            "total_entropy": 0,
            "total_time": 0,
            "total_entries": 0,
            "correct_count": 0,
        }
    )

    # Calculate statistics
    for entry in data:
        exit_number = entry["exit_number"]
        exit_stats[exit_number]["total_entropy"] += entry["entropy"]
        exit_stats[exit_number]["total_time"] += entry["time_taken"]
        exit_stats[exit_number]["total_entries"] += 1
        if entry["correct"]:
            exit_stats[exit_number]["correct_count"] += 1

    total_entries = len(data)
    total_correct = sum(stats["correct_count"] for stats in exit_stats.values())
    total_time_taken = sum(entry["time_taken"] for entry in data)

    # List to store refined statistics for each exit number
    refined_exit_stats = []

    # Generate statistics for each exit number
    for exit_number, stats in exit_stats.items():
        avg_entropy = stats["total_entropy"] / stats["total_entries"]
        avg_time_taken = stats["total_time"] / stats["total_entries"]
        accuracy = stats["correct_count"] / stats["total_entries"] * 100
        exit_percentage = stats["total_entries"] / total_entries * 100

        refined_exit_stats.append(
            {
                "Exit Number": exit_number,
                "Entries": stats["total_entries"],
                "Percentage": f"{exit_percentage:.2f}%",
                "Avg Entropy": avg_entropy,
                "Avg Time (ms)": avg_time_taken * 1000,
                "Accuracy (%)": f"{accuracy:.2f}%",
            }
        )

    # Write refined data into a new JSON file
    new_file_path = f"refined_{file_path}"
    with open(new_file_path, "w") as new_file:
        json.dump(refined_exit_stats, new_file, indent=4)

    print(f"Refined statistics for file '{file_path}' written to '{new_file_path}'")
    print()


# Get all JSON files in current directory
json_files = [file for file in os.listdir() if file.endswith(".json")]
json_files.sort()

# Process each JSON file
for file in json_files:
    process_json(file)
