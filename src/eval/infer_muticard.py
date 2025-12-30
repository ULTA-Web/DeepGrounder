import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
from PIL import Image as PILImage
import re
import cv2
import numpy as np
import wandb
from accelerate import Accelerator  # 引入Accelerate库


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str,
                        default="/mnt/dataset/zhangjiafan/Checkpoints/Qwen2.5-VL-3B-GRPO-RSG-lora8-train")
    parser.add_argument("--json_path", type=str, default="../../rec_jsons_processed/val.jsonl")
    parser.add_argument("--image_base_path", type=str, default="/mnt/dataset/zhangjiafan/RRSISD/data/images/JPEGImages")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    parser.add_argument("--result_images_dir", type=str, default="../outputs/box_QWEN")
    parser.add_argument("--wandb_run_name", type=str, default="qwen_lora8_grpo_all_val")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")  # 添加批量大小参数
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

def process_batch(batch_cases, processor, reasoning_model, args, accelerator, run_step_offset):
    try:
        # 准备批量输入
        messages, images, gt_bbox_list = [], [], []
        for case in batch_cases:
            image_path = os.path.join(args.image_base_path, case['image'])
            image = PILImage.open(image_path).convert("RGB")
            original_width, original_height = image.size
            resize_size = 840
            x_factor, y_factor = original_width / resize_size, original_height / resize_size

            QUESTION_TEMPLATE = "{} Output the one bbox for the most closely matched object in JSON format.You have to make sure that the target you're looking for is framed exactly in the bbox."

            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(case['problem'])}
                ]
            }])
            images.append((image_path, original_width, original_height))
            gt_bbox_list.append(case['solution'])

        # 批量处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        # 准备模型输入
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(accelerator.device)

        # 分布式生成
        with torch.no_grad():
            generated_ids = accelerator.unwrap_model(reasoning_model).generate(
                **inputs,
                use_cache=True,
                max_new_tokens=1024,
                do_sample=False
            )

        # 处理输出
        pred_bboxes = []
        for i, (out_ids, in_ids) in enumerate(zip(generated_ids, inputs.input_ids)):
            generated_ids_trimmed = out_ids[len(in_ids):]
            output_text = processor.batch_decode(
                [generated_ids_trimmed],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            image_path, orig_w, orig_h = images[i]
            x_factor = orig_w / 840
            y_factor = orig_h / 840
            pred_bbox = extract_bbox(output_text, x_factor, y_factor)
            pred_bboxes.append(pred_bbox)

            # 可视化（仅在主进程执行）
            if accelerator.is_main_process:
                img = cv2.imread(image_path)
                if pred_bbox:
                    x1, y1, x2, y2 = pred_bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "Pred", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if gt_bbox_list[i]:
                    x1, y1, x2, y2 = gt_bbox_list[i]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "GT", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                result_path = os.path.join(args.result_images_dir, f"{os.path.basename(image_path)}_result.jpg")
                cv2.imwrite(result_path, img)

        return pred_bboxes, gt_bbox_list

    except Exception as e:
        accelerator.print(f"Batch processing error: {str(e)}")
        return [], []


def main():
    args = parse_args()
    accelerator = Accelerator()  # 初始化Accelerator

    # 只在主进程初始化WandB
    if accelerator.is_main_process:
        wandb.init(
            project="visual-grounding-monitor",
            name=args.wandb_run_name,
            config={
                "model": os.path.basename(args.reasoning_model_path),
                "dataset": os.path.basename(args.json_path),
                "batch_size": args.batch_size * accelerator.num_processes  # 总批量大小
            },
        )

    # 加载模型
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.device},
    ).eval()

    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")

    # 使用accelerate准备模型
    reasoning_model = accelerator.prepare_model(reasoning_model)

    # 加载数据并分片
    with open(args.json_path, 'r') as f:
        all_cases = [json.loads(line) for line in f]

    # 分布式数据分片
    start_idx = accelerator.process_index * len(all_cases) // accelerator.num_processes
    end_idx = (accelerator.process_index + 1) * len(all_cases) // accelerator.num_processes
    local_cases = all_cases[start_idx:end_idx]

    pred_bboxes = []
    gt_bboxes = []

    # 分批处理
    for batch_start in range(0, len(local_cases), args.batch_size):
        batch_cases = local_cases[batch_start:batch_start + args.batch_size]

        try:
            batch_pred, batch_gt = process_batch(
                batch_cases,
                processor,
                reasoning_model,
                args,
                accelerator,
                run_step_offset=start_idx + batch_start
            )

            pred_bboxes.extend(batch_pred)
            gt_bboxes.extend(batch_gt)

        except Exception as e:
            accelerator.print(f"Error processing batch {batch_start}: {str(e)}")

    # 收集所有进程的结果
    all_preds = accelerator.gather_for_metrics(pred_bboxes)
    all_gts = accelerator.gather_for_metrics(gt_bboxes)

    # 只在主进程进行最终评估和保存
    if accelerator.is_main_process:
        final_metrics = evaluate_bboxes(all_preds, all_gts)
        if final_metrics:
            pr_table = wandb.Table(data=[
                [thresh, final_metrics[f'Pr@{thresh}']]
                for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]
            ], columns=["Threshold", "Precision"])

            wandb.log({
                "final/mean_iou": final_metrics['meanIoU'],
                "final/cum_iou": final_metrics['cumIoU'],
                "pr_curve": wandb.plot.line(pr_table, "Threshold", "Precision", title="Precision-Recall Curve")
            })

            with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
                json.dump(final_metrics, f, indent=4)

        wandb.finish()

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()