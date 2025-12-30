import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
from PIL import Image as PILImage
import re
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_root", type=str,
                        default="/mnt/dataset/zhangjiafan/Checkpoints/Qwen2.5-VL-3B-lora8-GRPO-dIOU0.5-Semantic-RSG",
                        help="包含checkpoint目录的根路径")  # ⭐新增参数
    parser.add_argument("--json_path", type=str, default="../../rec_jsons_processed/RRSISD/rrsisd_test.jsonl")
    parser.add_argument("--image_base_path", type=str, default="/mnt/dataset/zhangjiafan/RRSISD/data/images/JPEGImages")
    parser.add_argument("--output_root", type=str,
                        default="/mnt/dataset/zhangjiafan/Checkpoints/Qwen2.5-VL-3B-lora8-GRPO-dIOU0.5-Semantic-RSG",
                        help="所有checkpoint评估结果的根目录")  # ⭐修改参数
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
    # 输入格式：[x1, y1, x2, y2]
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
    metrics = {
        f'Pr@{thresh}': 0 for thresh in thresholds
    }
    metrics['meanIoU'] = 0
    metrics['cumIoU'] = 0
    metrics['count'] = len(pred_bboxes)

    total_intersect_area = 0
    total_union_area = 0
    iou_sum = 0

    for pred, gt in zip(pred_bboxes, gt_bboxes):
        intersect_area, union_area, iou = calculate_bbox_iou(pred, gt)
        total_intersect_area += intersect_area
        total_union_area += union_area
        iou_sum += iou

        for thresh in thresholds:
            metrics[f'Pr@{thresh}'] += iou >= thresh

    if metrics['count'] > 0:
        for thresh in thresholds:
            metrics[f'Pr@{thresh}'] /= metrics['count']
        metrics['meanIoU'] = iou_sum / metrics['count']
        metrics['cumIoU'] = total_intersect_area / total_union_area if total_union_area != 0 else 0

    return metrics

def process_single_case(case, processor, model, args):
    try:
        image_path = os.path.join(args.image_base_path, case['image'])
        image = PILImage.open(image_path).convert("RGB")
        original_width, original_height = image.size
        resize_size = 840
        x_factor, y_factor = original_width / resize_size, original_height / resize_size

        QUESTION_TEMPLATE = "{} Output the one bbox for the most closely matched object in JSON format.You have to make sure that the target you're looking for is framed exactly in the bbox."

        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)},
                {"type": "text", "text": QUESTION_TEMPLATE.format(case['problem'])}
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
            generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = \
        processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return extract_bbox(output_text, x_factor, y_factor), case['solution'], output_text
    except Exception as e:
        return None, None, str(e)


def evaluate_checkpoint(checkpoint_path, args):
    """⭐新增函数：评估单个checkpoint"""
    # 初始化模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(checkpoint_path, padding_side="left")

    # 加载测试数据
    with open(args.json_path, 'r') as f:
        test_cases = [json.loads(line) for line in f]

    # 执行推理
    pred_bboxes, gt_bboxes = [], []
    error_log = []

    for case in test_cases:
        try:
            pred_bbox, gt_bbox, _ = process_single_case(case, processor, model, args)
            if pred_bbox and gt_bbox:
                pred_bboxes.append(pred_bbox)
                gt_bboxes.append(gt_bbox)
        except Exception as e:
            error_log.append(f"{case['image']}\tError: {str(e)}")

    # 计算指标
    metrics = evaluate_bboxes(pred_bboxes, gt_bboxes)
    return metrics, error_log


def main():
    args = parse_args()

    # ⭐获取所有checkpoint目录（参考网页1、3）
    checkpoints = [
        d for d in os.listdir(args.checkpoints_root)
        if os.path.isdir(os.path.join(args.checkpoints_root, d))
           and d.startswith("checkpoint-")
    ]
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))  # 按数字排序

    all_metrics = {}

    for ckpt_dir in checkpoints:
        ckpt_path = os.path.join(args.checkpoints_root, ckpt_dir)
        output_dir = os.path.join(args.output_root, ckpt_dir)
        os.makedirs(output_dir, exist_ok=True)

        try:
            # ⭐初始化WandB（参考网页6、8）
            wandb_run = wandb.init(
                project="ckpt-test",
                name=ckpt_dir,
                config={
                    "checkpoint": ckpt_dir,
                    "model": os.path.basename(args.checkpoints_root)
                },
                reinit=True
            )

            # 执行评估
            metrics, errors = evaluate_checkpoint(ckpt_path, args)

            # ⭐保存结果（参考网页4、5）
            if metrics:
                # 保存指标
                with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
                    json.dump(metrics, f, indent=4)

                # 记录到总表
                all_metrics[ckpt_dir] = {
                    "meanIoU": metrics['meanIoU'],
                    "cumIoU": metrics['cumIoU'],
                    **{f"Pr@{thresh}": metrics[f'Pr@{thresh}'] for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]}
                }

                # WandB记录
                wandb.log({
                    "meanIoU": metrics['meanIoU'],
                    "cumIoU": metrics['cumIoU'],
                    **{f"Pr@{thresh}": metrics[f'Pr@{thresh}'] for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]}
                })

            # 保存错误日志
            if errors:
                with open(os.path.join(output_dir, "errors.log"), 'w') as f:
                    f.write("\n".join(errors))

            wandb_run.finish()

        except Exception as e:
            print(f"评估失败：{ckpt_dir}\n错误：{str(e)}")
            continue

    # ⭐保存汇总结果（参考网页6）
    with open(os.path.join(args.output_root, "all_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=4)


if __name__ == "__main__":
    main()
