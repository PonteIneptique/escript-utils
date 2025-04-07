import requests
from typing import List, Tuple, Dict, Optional
import itertools

import tqdm

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

def get_lines_and_regions(document_id:int, part_id: int, token: str) -> Tuple[List[Dict], List[Dict]]:
    """Fetch all lines for a part."""
    headers = {"Authorization": f"Token {token}"}
    url = f"{API_BASE}/documents/{document_id}/parts/{part_id}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    lines = data.get("lines", [])
    regions = data.get("regions", [])

    return lines, regions

def extract_bbox(mask: List[List[Dict]]) -> Tuple[int, int, int, int]:
    """Extracts a bounding box (x_min, y_min, x_max, y_max) from a mask polygon."""
    if isinstance(mask[0][0], dict):
        xs = [pt[0]["parsedValue"] for pt in mask]
        ys = [pt[1]["parsedValue"] for pt in mask]
    else:
        xs = [pt[0] for pt in mask]
        ys = [pt[1] for pt in mask]
    return min(xs), min(ys), max(xs), max(ys)

def rel_intersection(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    width_a = abs(box_a[0]-box_a[2])
    height_a = abs(box_a[1]-box_a[3])
    if width_a == 0 or height_a == 0:
        return 0

    inter_width = max(0, x_b - x_a)
    inter_height = max(0, y_b - y_a)
    relative_intersection = inter_width * inter_height / (width_a*height_a)

    return relative_intersection


def detect_reassignment(lines: List[Dict], regions: List[Dict]) -> List[Dict]:
    region_bboxes = {region["pk"]: extract_bbox(region["box"]) for region in regions if region.get("box")}
    require_updates: List[Dict] = []

    for line in lines:
        best_idx = -1
        best_intersection = 0
        for idx, zone in region_bboxes.items():
            current_intersection = rel_intersection(extract_bbox(line["mask"]), zone)
            if current_intersection > best_intersection and current_intersection > 0.0:
                best_intersection = current_intersection
                best_idx = idx
        if best_idx != -1 and best_idx != line.get("region"):
            line["region"] = best_idx
            require_updates.append(line)

    return require_updates


def bulk_update(document_id: int, part_id: int, token: str, lines: List[Dict]):
    url = f"{API_BASE}/documents/{document_id}/parts/{part_id}/lines/bulk_update/"
    headers = {"Authorization": f"Token {token}"}
    payload = {"lines": lines}
    response = requests.put(url, json=payload, headers=headers)
    if not response.ok:
        print(f"✗ Failed to bulk update for part {part_id}: {response.status_code}")
    return response


def recalculate_ordering(document_id: int, part_id: int, token: str) -> Optional[requests.Response]:
    """Sends a GET request to recalculate ordering for a single part."""
    url = f"{API_BASE}/documents/{document_id}/parts/{part_id}/recalculate_ordering/"
    headers = {"Authorization": f"Token {token}"}
    response = requests.post(url, headers=headers)
    if not response.ok:
        print(f"✗ Failed to recalculate for part {part_id}: {response.status_code}")
    return response

def main(document_id: int, token: str):
    print(f"Checking document {document_id} for lines to reassign...")
    part_ids = get_all_parts(document_id, token)

    with tqdm.tqdm(total=len(part_ids)) as bar:
        total_reassigned = 0
        for part_id in part_ids:
            lines, regions = get_lines_and_regions(document_id, part_id, token)
            to_reassign = detect_reassignment(lines, regions)

            if to_reassign:
                total_reassigned += len(to_reassign)
                bar.set_description(f"Reassigned: {total_reassigned} [+{len(to_reassign)} in {part_id}]")
                bulk_update(document_id, part_id, token, to_reassign)
                recalculate_ordering(document_id, part_id, token)
            bar.update(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect duplicate lines by IoU for an eScriptorium document.")
    parser.add_argument("document_id", type=int, help="Document ID on eScriptorium")
    parser.add_argument("token", type=str, help="Authorization token")

    args = parser.parse_args()
    main(args.document_id, args.token)
