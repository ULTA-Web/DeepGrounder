from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor,AutoModel
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
from PIL import Image
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import re

from torch import distributed as dist
from torchvision.transforms import ToTensor

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return ['max_pixels', 'min_pixels']
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    def format_reward(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
        
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union

        import math

        def diou(box1, box2):
            """
            计算两个边界框的距离交并比
            :param box1: 预测边界框，格式为 [x1, y1, x2, y2]
            :param box2: 真实边界框，格式为 [x1, y1, x2, y2]
            :return: DIoU 值
            """
            # 计算交集区域的左上角和右下角坐标
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2], box2[2])
            inter_y2 = min(box1[3], box2[3])
            # 计算交集区域的面积
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            else:
                inter_area = 0
            # 计算两个边界框的面积
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            # 计算并集区域的面积
            union_area = box1_area + box2_area - inter_area
            # 算两个边界框的中心点坐标
            box1_center_x = (box1[0] + box1[2]) / 2
            box1_center_y = (box1[1] + box1[3]) / 2
            box2_center_x = (box2[0] + box2[2]) / 2
            box2_center_y = (box2[1] + box2[3]) / 2
            # 计算两个中心点之间的距离
            center_distance = math.sqrt((box1_center_x - box2_center_x) ** 2 + (box1_center_y - box2_center_y) ** 2)
            # 计算包含两个边界框的最小外接矩形的对角线长度
            c_x1 = min(box1[0], box2[0])
            c_y1 = min(box1[1], box2[1])
            c_x2 = max(box1[2], box2[2])
            c_y2 = max(box1[3], box2[3])
            c_diagonal = math.sqrt((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2)
            # 计算 DIoU
            if union_area == 0:
                return 0
            iou_value = inter_area / union_area
            diou_value = iou_value - (center_distance ** 2) / (c_diagonal ** 2)
            return diou_value

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]

                        # if diou(bbox, sol) > 0.8:
                        #     reward = 5.0
                        # elif diou(bbox, sol) <= 0.8 and diou(bbox, sol) > 0.6:
                        #     reward = 4.0
                        # elif diou(bbox, sol) <= 0.6 and diou(bbox, sol) > 0.4:
                        #     reward = 3.0
                        # elif diou(bbox, sol) <= 0.4 and diou(bbox, sol) > 0.2:
                        #     reward = 2.0
                        # elif diou(bbox, sol) <= 0.2 and diou(bbox, sol) > 0.0:
                        #     reward = 1.0
                        # else:
                        #     reward = 0.0
                        reward = diou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards
    #
    # @staticmethod
    # def semantic_consistency_reward(completions, **kwargs):
    #     # 动态获取当前进程的local_rank
    #     local_rank = kwargs.get("local_rank", 0)
    #     device = torch.device(f"cuda:{local_rank}")
    #
    #     image_paths = kwargs["image_path"]
    #     normal_captions = kwargs["normal_caption"]
    #     processor = kwargs['processor']
    #     model = kwargs['model'].to(device)  # 访问底层模型
    #
    #     rewards = []
    #     answer_tag_pattern = r'<answer>(.*?)</answer>'
    #     bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    #     for completion, image_path, normal_caption in zip(completions, image_paths, normal_captions):
    #         content = completion[0]["content"]
    #         content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    #         if content_answer_match:
    #             content_answer = content_answer_match.group(1).strip()
    #             bbox_match = re.search(bbox_pattern, content_answer)
    #             if bbox_match:
    #                 bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)),
    #                         int(bbox_match.group(4))]
    #                 try:
    #                     image = Image.open(image_path)
    #                     cropped_image = ToTensor()(image.crop(bbox)).to(device)
    #                     text = normal_caption  # 使用数据集中的 normal_caption 作为匹配文本描述
    #                     inputs = processor(text=text, images=cropped_image, padding="max_length",
    #                                             return_tensors="pt").to(device)
    #                     model.eval()
    #                     with torch.no_grad():
    #                         outputs = model(**inputs)
    #                     logits_per_image = outputs.logits_per_image
    #                     probs = torch.sigmoid(logits_per_image)
    #                     reward = probs.item()
    #                 except Exception as e:
    #                     print(f"Error computing semantic consistency reward: {e}")
    #                     reward = 0.0
    #             else:
    #                 reward = 0.0
    #         else:
    #             reward = 0.0
    #         rewards.append(reward)
    #     return rewards

    # @staticmethod
    # def weighted_rewards(completions, solution, image_paths, normal_captions, weight_format=0.2, weight_iou=0.5,
    #                      weight_semantic=0.3, **kwargs):
    #     module = Qwen2VLModule()
    #     format_rewards = Qwen2VLModule.format_reward_rec(completions, **kwargs)
    #     iou_rewards = Qwen2VLModule.iou_reward(completions, solution, **kwargs)
    #     semantic_rewards = module.semantic_consistency_reward(completions, solution, image_paths, normal_captions,
    #                                                           **kwargs)
    #
    #     weighted_rewards = []
    #     for format_reward, iou_reward, semantic_reward in zip(format_rewards, iou_rewards, semantic_rewards):
    #         weighted_reward = weight_format * format_reward + weight_iou * iou_reward + weight_semantic * semantic_reward
    #         weighted_rewards.append(weighted_reward)
    #
    #     return weighted_rewards