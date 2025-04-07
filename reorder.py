import requests
import tqdm
from typing import List, Optional

API_BASE = "https://escriptorium.inria.fr/api"

def get_all_part_ids(document_id: int, token: str) -> List[int]:
    """Fetches all part IDs for a given document, following pagination."""
    headers = {"Authorization": f"Token {token}"}
    url = f"{API_BASE}/documents/{document_id}/parts/"
    part_ids = []

    while url:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        part_ids.extend(part["pk"] for part in data.get("results", []))
        url = data.get("next")

    return part_ids

def recalculate_ordering(document_id: int, part_id: int, token: str) -> Optional[requests.Response]:
    """Sends a GET request to recalculate ordering for a single part."""
    url = f"{API_BASE}/documents/{document_id}/parts/{part_id}/recalculate_ordering/"
    headers = {"Authorization": f"Token {token}"}
    response = requests.post(url, headers=headers)
    if not response.ok:
        print(f"✗ Failed to recalculate for part {part_id}: {response.status_code}")
    return response

def recalculate_all_parts(document_id: int, token: str):
    """Fetches all part IDs and sends recalculate ordering requests."""
    print(f"Fetching parts for document {document_id}...")
    part_ids = get_all_part_ids(document_id, token)
    print(f"Found {len(part_ids)} parts. Recalculating ordering...")

    for part_id in tqdm.tqdm(part_ids):
        recalculate_ordering(document_id, part_id, token)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recalculate ordering for all parts of a document.")
    parser.add_argument("document_id", type=int, help="Document ID on eScriptorium")
    parser.add_argument("token", type=str, help="Authorization token for eScriptorium API")

    args = parser.parse_args()

    recalculate_all_parts(args.document_id, args.token)
