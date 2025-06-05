def parse_item_ids(file_path, unique=False, skip_header=True):
    """
    Parses item_ids from the atomic-format MovieLens file.

    Args:
        file_path (str): Path to the atomic-format movie metadata file.
        unique (bool): If True, returns a set of unique item_ids.
        skip_header (bool): If True, skips the first header line.

    Returns:
        List[int] or Set[int]: Parsed item_ids.
    """
    item_ids = set() if unique else []

    with open(file_path, "r", encoding="utf-8") as f:
        if skip_header:
            next(f)

        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0].isdigit():
                item_id = int(parts[0])
                if unique:
                    item_ids.add(item_id)
                else:
                    item_ids.append(item_id)

    return item_ids


file_path = r"venv\Lib\site-packages\recbole\dataset_example\ml-100k\ml-100k.item"


# Get all item_ids (including duplicates, if any)
item_ids = parse_item_ids(file_path)

# Get unique item_ids
unique_item_ids = parse_item_ids(file_path, unique=True)

print(f"Total item entries: {len(item_ids)}")
print(f"Unique item count: {len(unique_item_ids)}")
