
import json
import os
import csv
import re

# Define paths for the files (use relative paths based on your structure)
SYSTEM_METADATA_FILE = "./cache/embedding-3-50/metadata_cache.json"
SYSTEM_EMBEDDING_FILE = "./cache/embedding-3-50/embedding_vectors.json"
CLEAN_DIR = "./cache/embedding-3-50"
CLEANED_DIR = f"{CLEAN_DIR}-cleaned"
LOG_FILE = "{CLEAN_DIR}/deleted_short_sentences.csv"  # CSV file to log deleted entries

# Ensure the cleaned directory exists
os.makedirs(CLEANED_DIR, exist_ok=True)

def load_json(file_path):
    """Helper function to load JSON data from a file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"File {file_path} does not exist.")
        return {}

def save_json(file_path, data):
    """Helper function to save data as JSON to a file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {file_path}.")

def log_deleted_entries(deleted_entries):
    """Log deleted entries (sentence, article, url) to a CSV file."""
    with open(LOG_FILE, mode='w', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "article", "url"])  # Write header
        for entry in deleted_entries:
            writer.writerow([entry["sentence"], entry["article"], entry["url"]])
    print(f"Deleted entries have been logged to {LOG_FILE}.")


def clean_and_count_length(sentence):
    """
    1. 删除指定标点符号
    2. 删除所有英文字母
    3. 返回剩余字符长度
    """
    # 删除指定标点符号
    sentence = re.sub(r"[.\“，。/[\]{}【】\-——~]", "", sentence)

    # 删除所有英文字母（不区分大小写）
    sentence = re.sub(r"[a-zA-Z]", "", sentence)

    # 删除空格和制表符
    sentence = sentence.strip()

    return len(sentence)

def clean_short_sentences():
    # Load the system metadata and embeddings
    system_metadata = load_json(SYSTEM_METADATA_FILE)
    system_embeddings = load_json(SYSTEM_EMBEDDING_FILE)
    
    # Keep track of the sentences to remove and log
    to_remove = []
    deleted_entries = []

    # Find entries where the cleaned sentence length is less than 10
    for key, entry in system_metadata.items():
        sentence = entry.get("sentence", "")
        if clean_and_count_length(sentence) < 10:  # Check if the cleaned sentence is too short
            to_remove.append(key)
            deleted_entries.append({
                "sentence": sentence,
                "article": entry.get("article", "Unknown"),
                "url": entry.get("url", "Unknown")
            })

    # Remove entries from both system_metadata and system_embeddings
    for key in to_remove:
        if key in system_metadata:
            del system_metadata[key]
        if key in system_embeddings:
            del system_embeddings[key]
    
    # Log deleted entries to CSV
    if deleted_entries:
        log_deleted_entries(deleted_entries)

    # Save the cleaned data to the cleaned directory
    cleaned_metadata_path = os.path.join(CLEANED_DIR, "system_metadata_cache.json")
    cleaned_embeddings_path = os.path.join(CLEANED_DIR, "system_embedding_vectors.json")
    
    save_json(cleaned_metadata_path, system_metadata)
    save_json(cleaned_embeddings_path, system_embeddings)

    print(f"Removed {len(to_remove)} entries from the system metadata and embeddings.")
    print(f"Cleaned data saved in {CLEANED_DIR}")

if __name__ == "__main__":
    clean_short_sentences()
