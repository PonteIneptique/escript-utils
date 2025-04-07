import requests
from typing import List, Tuple, Dict, Optional
import itertools

API_BASE = "https://escriptorium.inria.fr/api"

def get_all_parts(document_id: int, token: str) -> List[int]:
    """Fetch all part IDs for a document."""
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

def get_lines(document_id:int, part_id: int, token: str) -> List[Dict]:
    """Fetch all lines for a part."""
    headers = {"Authorization": f"Token {token}"}
    url = f"{API_BASE}/documents/{document_id}/parts/{part_id}"
    lines = []

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    lines.extend(data.get("lines", []))

    return lines

def extract_bbox(mask: List[List[Dict]]) -> Tuple[int, int, int, int]:
    """Extracts a bounding box (x_min, y_min, x_max, y_max) from a mask polygon."""
    if isinstance(mask[0][0], dict):
        xs = [pt[0]["parsedValue"] for pt in mask]
        ys = [pt[1]["parsedValue"] for pt in mask]
    else:
        xs = [pt[0] for pt in mask]
        ys = [pt[1] for pt in mask]
    return min(xs), min(ys), max(xs), max(ys)

def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    if inter_area == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return inter_area / union_area


def detect_duplicates_to_delete(lines: List[Dict]) -> List[int]:
    bboxes = {line["pk"]: extract_bbox(line["mask"]) for line in lines if line.get("mask")}
    to_delete = set()
    checked_pairs = set()

    for (id1, box1), (id2, box2) in itertools.combinations(bboxes.items(), 2):
        pair_key = tuple(sorted((id1, id2)))
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)
        score = iou(box1, box2)
        if score > 0.5:
            # Delete the line with the smaller x_min
            delete_id = id1 if box1[0] < box2[0] else id2
            to_delete.add(delete_id)

    return list(to_delete)

def bulk_delete_lines(document_id: int, part_id: int, line_ids: List[int], token: str):
    url = f"{API_BASE}/documents/{document_id}/parts/{part_id}/lines/bulk_delete/"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json"
    }
    payload = {"lines": line_ids}
    response = requests.post(url, json=payload, headers=headers)

    if response.ok:
        print(f"✓ Deleted {len(line_ids)} lines in part {part_id}")
    else:
        print(f"✗ Failed to delete lines in part {part_id}: {response.status_code} - {response.text}")

def recalculate_ordering(document_id: int, part_id: int, token: str) -> Optional[requests.Response]:
    """Sends a GET request to recalculate ordering for a single part."""
    url = f"{API_BASE}/documents/{document_id}/parts/{part_id}/recalculate_ordering/"
    headers = {"Authorization": f"Token {token}"}
    response = requests.post(url, headers=headers)
    if not response.ok:
        print(f"✗ Failed to recalculate for part {part_id}: {response.status_code}")
    return response

def main(document_id: int, token: str):
    print(f"Checking document {document_id} for duplicate lines...")
    part_ids = get_all_parts(document_id, token)

    for part_id in part_ids:
        lines = get_lines(document_id, part_id, token)
        to_delete = detect_duplicates_to_delete(lines)

        if to_delete:
            print(f"\nPart {part_id}: {len(to_delete)} lines to delete due to duplication.")
            bulk_delete_lines(document_id, part_id, to_delete, token)
            recalculate_ordering(document_id, part_id, token)
        else:
            print(f"Part {part_id} has no duplicates to delete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect duplicate lines by IoU for an eScriptorium document.")
    parser.add_argument("document_id", type=int, help="Document ID on eScriptorium")
    parser.add_argument("token", type=str, help="Authorization token")

    args = parser.parse_args()
    main(args.document_id, args.token)
