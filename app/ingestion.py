"""
app/ingestion.py
Handles document discovery and parsing across the data directory.
"""

import os
from app.config import settings
from app.hr_loader import load_employees, save_employees
from app.markdown import convert_document


def get_data_folders(data_dir: str = settings.data_dir) -> list[str]:
    """Return names of all immediate subdirectories inside data_dir, excluding qdrant_local."""
    return [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f)) and f != "qdrant_local"
    ]




def ingest_all(data_dir: str = settings.data_dir) -> dict:
    """
    Walk every subfolder in data_dir and process each file:
      - .md  files → skipped (already markdown)
      - .csv files → loaded as employee data and saved
      - all others → parsed to markdown via docling

    Returns:
        {
            "documents": {filename: markdown_str, ...},
            "employees": [employee_dict, ...]
        }
    """
    folders = get_data_folders(data_dir)
    print(f"Found folders: {folders}\n")

    documents = {}
    employees = []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) 
            and not f.startswith(".")
            and not f.startswith("~$")
        ]

        for file in files:
            file_path = os.path.join(folder_path, file)

            if file.endswith(".md"):
                print(f"Reading [{folder}] -> {file} (already markdown, skipping conversion)")
                with open(file_path, encoding="utf-8") as md_file:
                    documents[file] = {
                        "markdown": md_file.read(),
                        "doc_obj":  None,
                        "collection": folder,
                    }
                continue

            if file.endswith(".csv"):
                print(f"Loading employee data from [{folder}] -> {file}")
                employees = load_employees(file_path)
                print(f"Found {len(employees)} employees\n")
                for emp in employees:
                    print(f"  [{emp['access_role']:12}] {emp['name']:25} | {emp['role']:30} | {emp['department']} ({emp['designation_level']})")
                save_employees(employees)
                continue

            print(f"\nParsing [{folder}] -> {file}")
            markdown, doc_obj = convert_document(file_path)
            documents[file] = {
                "markdown":   markdown,
                "doc_obj":    doc_obj,
                "collection": folder,
            }
            print(markdown)

    return {"documents": documents, "employees": employees}
