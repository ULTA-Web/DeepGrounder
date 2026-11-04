"""Convert official DIOR-RSVG XML into SFT, GRPO and evaluation annotations.

Split IDs are zero-based OBJECT/EXPRESSION indices, as in the official loader,
not image IDs. XML paths are sorted before enumerating objects.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    split_ids = {}
    for split in ("train", "val", "test"):
        ids = [int(line.strip()) for line in (root / f"{split}.txt").read_text().splitlines() if line.strip()]
        if len(ids) != len(set(ids)) or any(i < 0 for i in ids):
            raise ValueError(f"{split}: duplicate or negative expression indices")
        split_ids[split] = set(ids)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if split_ids[a] & split_ids[b]:
            raise ValueError(f"Overlapping expression indices in {a} and {b}")

    records = []
    xml_paths = sorted((root / "Annotations").rglob("*.xml"))
    if not xml_paths:
        raise ValueError("No XML files found under Annotations")
    for xml_path in xml_paths:
        xml = ET.parse(xml_path).getroot()
        filename = xml.findtext("filename")
        if not filename:
            raise ValueError(f"Missing filename in {xml_path}")
        # Match the original loader; downloaded annotations refer to JPEGImages.
        image_path = root / "JPEGImages" / filename
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        for obj in xml.findall("object"):
            box_node = obj.find("bndbox")
            if box_node is None and len(obj) > 2:
                box_node = obj[2]  # official loader uses member[2]
            if box_node is None:
                raise ValueError(f"Missing bounding box in {xml_path}")
            box = [int(float(box_node.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")]
            if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Invalid xyxy box in {xml_path}: {box}")
            # The official loader uses member[3] for the referring expression.
            if len(obj) < 4 or not (obj[3].text or "").strip():
                raise ValueError(f"Missing expression in {xml_path}")
            records.append({"image": filename, "caption": obj[3].text.strip(), "bbox": box})

    all_ids = set().union(*split_ids.values())
    if not all_ids or max(all_ids) >= len(records):
        raise ValueError("Split IDs do not match the sorted XML object enumeration")
    if all_ids != set(range(len(records))):
        raise ValueError("Split files must partition all XML expressions; verify the dataset release")

    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, ids in split_ids.items():
        chosen = [(i, records[i]) for i in sorted(ids)]
        counts[split] = len(chosen)
        grpo, evaluation, sft = [], [], []
        for i, record in chosen:
            caption, image, box = record["caption"], record["image"], record["bbox"]
            problem = f"Please provide the bounding box coordinate of the region this sentence describes: {caption}"
            grpo.append({"id": i, "image": image, "problem": problem,
                         "normal_caption": caption, "solution": box})
            evaluation.append({"id": i, "file_name": image, "sent": caption, "bbox": box})
            answer = json.dumps([{"bbox_2d": box, "label": caption}], ensure_ascii=False)
            sft.append({"messages": [{"role": "user", "content": "<image>" + problem},
                                      {"role": "assistant", "content": answer}],
                        "images": [str(root / "JPEGImages" / image)]})
        for suffix, rows in (("grpo", grpo), ("eval", evaluation)):
            with (output / f"{split}_{suffix}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output / f"{split}_sft.json").write_text(json.dumps(sft, ensure_ascii=False, indent=2), encoding="utf-8")

    # JSON is valid YAML, and quotes Windows paths safely.
    (output / "dior_grpo.yaml").write_text(
        json.dumps({"datasets": [{"json_path": str(output / "train_grpo.jsonl")}]}, indent=2),
        encoding="utf-8")
    print(json.dumps({"expression_counts": counts, "output": str(output)}, indent=2))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/DIOR-RSVG"))
    args = parser.parse_args()
    prepare(args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
