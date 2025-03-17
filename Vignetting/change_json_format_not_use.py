import os
import json
import numpy as np


def process_json_files(directory):
    """遍历当前目录下的所有 JSON 文件并进行转换"""
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)

                if isinstance(data, dict):
                    values = np.array(list(data.values()), dtype=np.float16)
                    converted_data = dict(zip(data.keys(), values.tolist()))

                    with open(filepath, "w", encoding="utf-8") as file:
                        json.dump(converted_data, file, ensure_ascii=False, indent=4)
                    print(f"Processed: {filename}")
                else:
                    print(f"Skipped (not a dict): {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    process_json_files(os.getcwd())