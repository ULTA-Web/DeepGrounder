import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import os
from PIL import Image as PILImage
import re
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--reasoning_model_path", type=str,default="//wsl.localhost/Ubuntu-24.04/mnt/dataset/zhangjiafan/Checkpoints/DIOR_RSVG/GRPO-dIOU0.5-Semantic-DIOR")
    parser.add_argument("--reasoning_model_path", type=str, default="G:/model_cache/checkpoint-4500-best")
    # parser.add_argument("--json_path", type=str, default="../../rec_jsons_processed/test.jsonl")
    # parser.add_argument("--image_base_path", type=str, default="D:/Data/RRSIS-D/images/rrsisd/JPEGImages")

    parser.add_argument("--json_path", type=str, default="E:/VLM-R1-main/rec_jsons_processed/rrsisd_test.jsonl")
    parser.add_argument("--image_base_path", type=str, default="G:/data/RRSISD/JPEGImages")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument("--result_images_dir", type=str, default="../outputs/box_QWEN")
    return parser.parse_args()


def extract_bbox(output_text, x_factor, y_factor):
    json_pattern = r'{[^}]+}'
    json_match = re.search(json_pattern, output_text)
    content_bbox = None

    if json_match:
        try:
            data = json.loads(json_match.group(0))
            bbox_key = next((key for key in data.keys() if 'bbox' in key.lower()), None)
            if bbox_key and len(data[bbox_key]) == 4:
                content_bbox = [round(int(x) * x_factor) for x in data[bbox_key]]
        except:
            pass
    return content_bbox


def calculate_bbox_iou(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    intersect_x_min = max(x1_min, x2_min)
    intersect_y_min = max(y1_min, y2_min)
    intersect_x_max = min(x1_max, x2_max)
    intersect_y_max = min(y1_max, y2_max)

    intersect_area = max(0, intersect_x_max - intersect_x_min) * max(0, intersect_y_max - intersect_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersect_area

    return intersect_area, union_area, intersect_area / union_area if union_area != 0 else 0.0


def evaluate_bboxes(pred_bboxes, gt_bboxes):
    if not pred_bboxes:
        return None

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    metrics = {f'Pr@{thresh}': 0 for thresh in thresholds}
    metrics['oIOU'] = 0  # Overall IoU
    metrics['mIOU'] = 0  # Mean IoU

    total_intersect = 0
    total_union = 0
    iou_sum = 0
    valid_count = 0

    for pred, gt in zip(pred_bboxes, gt_bboxes):
        intersect, union, iou = calculate_bbox_iou(pred, gt)
        print("iou:", iou)
        total_intersect += intersect
        total_union += union
        iou_sum += iou
        valid_count += 1

        for thresh in thresholds:
            metrics[f'Pr@{thresh}'] += iou >= thresh

    if valid_count > 0:
        for thresh in thresholds:
            metrics[f'Pr@{thresh}'] /= valid_count
        metrics['mIOU'] = iou_sum / valid_count
        metrics['oIOU'] = total_intersect / total_union if total_union != 0 else 0

    return metrics


def process_single_case(case, processor, reasoning_model, args):
    # try:
        image_path = os.path.join(args.image_base_path, case['file_name'])
        image = PILImage.open(image_path).convert("RGB")
        original_width, original_height = image.size
        resize_size = 840
        x_factor, y_factor = original_width / resize_size, original_height / resize_size

        # QUESTION_TEMPLATE = "{} Output the one bbox for the most closely matched object in JSON format.You have to make sure that the target you're looking for is framed exactly in the bbox."


        QUESTION_TEMPLATE = (
            "<image>\n"
            "Locate the target described in '{Question}' within the remote sensing image by following a spatial reasoning process.\n"
            "Your response must include:\n"
            "1. Feature Extraction: Identify key visual features (e.g., shape, color, size, texture) and contextual location clues from the question.\n"
            "2. Spatial Positioning: Pinpoint candidate regions using precise spatial terms (e.g., 'upper-left quadrant', 'near riverbank') and analyze their features against the question.\n"
            "3. Contextual Contrast: Compare candidate regions with similar objects in adjacent areas to rule out mismatches (e.g., 'distinguish between lake and pond based on shape and size in the southwest corner').\n"
            "4. Position Justification: Confirm the final location by explaining why it best matches all features and spatial context.\n"
            "Output the reasoning in  tags and the bounding box in JSON format.\n"
            "i.e., <think> thinking process here </think> <answer>{Answer}</answer>"
        )
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)},
                {"type": "text", "text": QUESTION_TEMPLATE.format(case['sent'])}
            ]
        }]

        text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)]
        image_inputs, video_inputs = process_vision_info([message])
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = \
        processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return extract_bbox(output_text, x_factor, y_factor), case['bbox']
    # except Exception as e:
    #     return None, None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.result_images_dir, exist_ok=True)

    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).cuda().eval()

    # import torch
    # from modelscope import AutoTokenizer, AutoModel
    # path = "OpenGVLab/InternVL3-8B"
    # reasoning_model = AutoModel.from_pretrained(
    #     path,
    #     torch_dtype=torch.bfloat16,
    #     low_cpu_mem_usage=True,
    #     use_flash_attn=True,
    #     trust_remote_code=True).eval().cuda()

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left",trust_remote_code=True)

    with open(args.json_path, 'r') as f:
        test_cases = [json.loads(line) for line in f]

    pred_bboxes = []
    gt_bboxes = []
    error_count = 0

    with open(os.path.join(args.output_dir, "error_cases.txt"), "w") as err_f:
        for case in tqdm(test_cases):
            # try:
                pred, gt = process_single_case(case, processor, reasoning_model, args)
                print(pred, gt)
                if pred and gt:
                    pred_bboxes.append(pred)
                    gt_bboxes.append(gt)
            # except Exception as e:
            #     error_count += 1
            #     err_f.write(f"{case['file_name']}\tError: {str(e)}\n")

    metrics = evaluate_bboxes(pred_bboxes, gt_bboxes)

    # 打印最终结果
    print("\nEvaluation Results:")
    print(f"Test Samples: {len(test_cases)}")
    print(f"Valid Samples: {len(pred_bboxes)}")
    print(f"Error Samples: {error_count}\n")

    if metrics:
        print("Metric\t\tValue")
        print("-----------------------")
        for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
            print(f"Pr@{thresh}\t\t{metrics[f'Pr@{thresh}']:.4f}")
        print(f"oIOU\t\t{metrics['oIOU']:.4f}")
        print(f"mIOU\t\t{metrics['mIOU']:.4f}")

        # 保存结果到文件
        with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()