# import re
# import json
# import logging
# from PIL import Image
# from torch import nn
# from transformers import AutoModel, AutoProcessor
# import torch
#
# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class CLIPFunction():
#     def __init__(self, model_name="lcybuaa/Git-RSCLIP", cache_dir="/mnt/dataset/zhangjiafan/huggingface"):
#         super().__init__()
#         try:
#             # 加载RSCLIP模型和处理器
#             self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
#             self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
#             self.device = self.model.device
#
#             # 直接从处理器获取期望的图像大小并转换为元组
#             self._get_image_size()
#
#             logger.info(f"Loaded RSCLIP model on device: {self.device}")
#             logger.info(f"Expected image size: {self.image_size}")
#
#         except Exception as e:
#             logger.error(f"Failed to initialize model: {e}")
#             raise
#
#     def _get_image_size(self):
#         """从处理器获取图像尺寸并转换为元组格式"""
#         # 尝试从图像处理器获取尺寸
#         if hasattr(self.processor, 'image_processor') and hasattr(self.processor.image_processor, 'size'):
#             size_dict = self.processor.image_processor.size
#             if isinstance(size_dict, dict):
#                 # 字典格式 {'height': H, 'width': W} -> 转换为元组 (W, H)
#                 self.image_size = (size_dict.get('width', 256), size_dict.get('height', 256))
#             elif isinstance(size_dict, (tuple, list)):
#                 # 已经是元组或列表格式
#                 self.image_size = tuple(size_dict)
#             else:
#                 # 未知格式，使用默认值
#                 self.image_size = (256, 256)
#         else:
#             # 没有图像处理器信息，使用常见CLIP模型默认尺寸
#             self.image_size = (256, 256)
#
#     def __call__(self, prompts=None, completions=None, **kwargs):
#         """
#         使类实例可调用，适配训练代码中的调用方式
#         """
#         # 1. 从 kwargs 中提取 image_paths 和 normal_captions
#         image_paths = kwargs.get("image_paths", [])
#         normal_captions = kwargs.get("normal_captions", [])
#         print("image_paths", image_paths)
#         print("normal_captions", normal_captions)
#
#         # 2. 参数验证和默认值处理
#         if completions is None:
#             completions = []
#
#         num_completions = len(completions)
#
#         # 如果参数为空，创建默认列表
#         if not image_paths:
#             logger.warning("No image_paths provided, using empty list")
#             image_paths = [None] * num_completions
#
#         if not normal_captions:
#             logger.warning("No normal_captions provided, using empty strings")
#             normal_captions = [""] * num_completions
#
#         # 3. 确保所有列表长度一致
#         if len(image_paths) != num_completions:
#             logger.warning(f"Adjusting image_paths length from {len(image_paths)} to {num_completions}")
#             if len(image_paths) > num_completions:
#                 image_paths = image_paths[:num_completions]
#             else:
#                 image_paths = image_paths + [image_paths[-1] if image_paths else None] * (num_completions - len(image_paths))
#
#         if len(normal_captions) != num_completions:
#             logger.warning(f"Adjusting normal_captions length from {len(normal_captions)} to {num_completions}")
#             if len(normal_captions) > num_completions:
#                 normal_captions = normal_captions[:num_completions]
#             else:
#                 normal_captions = normal_captions + [normal_captions[-1] if normal_captions else ""] * (num_completions - len(normal_captions))
#
#         logger.info(f"Processing {num_completions} completions with {len(image_paths)} images and {len(normal_captions)} captions")
#
#         # 4. 调用forward方法
#         return self.forward(completions, image_paths, normal_captions)
#
#     @torch.no_grad()
#     def forward(self, completions, image_paths, normal_captions):
#         """
#         计算裁剪区域与文本描述的语义一致性奖励
#         """
#         self.model.eval()
#         rewards = []
#
#         # 用于统计
#         total_samples = len(completions)
#         processed = 0
#
#         logger.info(f"Processing {total_samples} samples...")
#
#         for i, (completion, image_path, caption) in enumerate(zip(completions, image_paths, normal_captions)):
#             # try:
#             # 1. 提取助手回复内容
#             if isinstance(completion, list) and completion and isinstance(completion[0], dict):
#                 content = completion[0].get("content", "")
#             else:
#                 content = str(completion)
#
#             # 2. 尝试从内容中提取<answer>标签
#             print("content:", content)
#             answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
#             if not answer_match:
#                 logger.warning(f"No <answer> tag found in sample {i}")
#                 rewards.append(0.0)
#                 continue
#
#             answer_content = answer_match.group(1).strip()
#
#             # 3. 尝试解析边界框
#             bbox = None
#
#             # 尝试1: 解析JSON格式
#             try:
#                 json_match = re.search(r'\{.*\}', answer_content, re.DOTALL)
#                 if json_match:
#                     json_str = json_match.group(0)
#                     bbox_data = json.loads(json_str)
#
#                     if isinstance(bbox_data, dict) and "bbox_2d" in bbox_data:
#                         bbox = bbox_data["bbox_2d"]
#                     elif isinstance(bbox_data, list) and bbox_data and isinstance(bbox_data[0], dict):
#                         if "bbox_2d" in bbox_data[0]:
#                             bbox = bbox_data[0]["bbox_2d"]
#             except:
#                 pass
#
#             # 尝试2: 解析数组格式
#             if bbox is None:
#                 list_match = re.search(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', answer_content)
#                 if list_match:
#                     bbox = [int(list_match.group(1)), int(list_match.group(2)),
#                             int(list_match.group(3)), int(list_match.group(4))]
#
#             # 4. 验证边界框
#             if not bbox or len(bbox) != 4 or not all(isinstance(c, (int, float)) for c in bbox):
#                 logger.warning(f"Invalid bbox in sample {i}")
#                 rewards.append(0.0)
#                 continue
#
#             print("bbox:", bbox)
#             # 5. 处理图像
#             image = Image.open(image_path).convert("RGB")
#             width, height = image.size
#
#             # 确保边界框在图像范围内
#             x1, y1, x2, y2 = [int(c) for c in bbox]
#             x1 = max(0, min(x1, width - 1))
#             y1 = max(0, min(y1, height - 1))
#             x2 = max(0, min(x2, width - 1))
#             y2 = max(0, min(y2, height - 1))
#
#             if x1 >= x2 or y1 >= y2:
#                 logger.warning(f"Invalid bbox coordinates in sample {i}")
#                 rewards.append(0.0)
#                 continue
#
#             # 裁剪并调整到模型期望的大小 (使用元组格式)
#             cropped_img = image.crop((x1, y1, x2, y2)).resize(self.image_size)
#             print("text:",caption)
#             print("cropped_img:",cropped_img.size)
#
#             # 6. 使用处理器准备输入
#             inputs = self.processor(
#                 text=[caption],
#                 images=cropped_img,
#                 padding="max_length",
#                 return_tensors="pt"
#             ).to(self.device)
#
#             # 确保像素值张量维度正确
#             # if inputs.pixel_values.dim() != 4:
#             #     inputs.pixel_values = inputs.pixel_values.unsqueeze(0)
#             # 在模型推理前确保输入维度正确
#             if inputs.pixel_values.dim() == 3:  # 若为 [C, H, W]
#                 inputs.pixel_values = inputs.pixel_values.unsqueeze(0)  # 转为 [1, C, H, W]
#
#
#             inputs['pixel_values'] = inputs['pixel_values'].clone().detach().contiguous()
#             logger.info(f"Input shape: {inputs.pixel_values.shape}")
#             # 7. 模型推理
#             outputs = self.model(**inputs)
#
#             # 8. 处理输出
#             if hasattr(outputs, 'logits_per_image'):
#                 logits = outputs.logits_per_image
#             elif hasattr(outputs, 'logits'):
#                 logits = outputs.logits
#             else:
#                 # 尝试获取第一个输出
#                 if isinstance(outputs, tuple) and len(outputs) > 0:
#                     logits = outputs[0]
#                 else:
#                     logits = outputs
#
#             # 确保正确的维度
#             if logits.dim() < 2:
#                 logits = logits.unsqueeze(0)
#
#             # 计算概率分数
#             probs = torch.sigmoid(logits)
#
#             # 提取奖励值
#             if probs.dim() == 2:
#                 reward = probs[0, 0].item()
#             else:
#                 reward = probs.item()
#
#             rewards.append(reward)
#             processed += 1
#             logger.info(f"Sample {i} reward: {reward:.4f}")
#
#             # except Exception as e:
#             #     logger.error(f"Error processing sample {i}: {e}")
#             #     rewards.append(0.0)
#
#         logger.info(f"Successfully processed {processed}/{total_samples} samples")
#         return rewards
#
#


# import re
# import json
# import logging
# from PIL import Image
# from torch import nn
# from transformers import AutoModel, AutoProcessor
# import torch
#
# # 配置日志
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
#
# class CLIPFunction():
#     def __init__(self, model_name="google/siglip-base-patch16-256", cache_dir="/mnt/dataset/zhangjiafan/huggingface"):
#         super().__init__()
#         try:
#             # 加载RSCLIP模型和处理器
#             self.conv = nn.Conv2d(3,1024,16,16).cuda()
#             self.clip_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
#             self.clip_processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
#             self.device = self.clip_model.device
#
#             # 直接从处理器获取期望的图像大小并转换为元组
#             self._get_image_size()
#
#             logger.info(f"Loaded RSCLIP model on device: {self.device}")
#             logger.info(f"Expected image size: {self.image_size}")
#
#         except Exception as e:
#             logger.error(f"Failed to initialize model: {e}")
#             raise
#
#     def _get_image_size(self):
#         """从处理器获取图像尺寸并转换为元组格式"""
#         # 尝试从图像处理器获取尺寸
#         if hasattr(self.clip_processor, 'image_processor') and hasattr(self.clip_processor.image_processor, 'size'):
#             size_dict = self.clip_processor.image_processor.size
#             if isinstance(size_dict, dict):
#                 # 字典格式 {'height': H, 'width': W} -> 转换为元组 (W, H)
#                 self.image_size = (size_dict.get('width', 256), size_dict.get('height', 256))
#             elif isinstance(size_dict, (tuple, list)):
#                 # 已经是元组或列表格式
#                 self.image_size = tuple(size_dict)
#             else:
#                 # 未知格式，使用默认值
#                 self.image_size = (256, 256)
#         else:
#             # 没有图像处理器信息，使用常见CLIP模型默认尺寸
#             self.image_size = (256, 256)
#
#     @torch.no_grad()
#     def forward(self, completions, image_paths, normal_captions):
#         """
#         计算裁剪区域与文本描述的语义一致性奖励
#         """
#         self.clip_model.eval()
#         rewards = []
#
#         # 用于统计
#         total_samples = len(completions)
#         processed = 0
#
#         logger.info(f"Processing {total_samples} samples...")
#
#         for i, (completion, image_path, caption) in enumerate(zip(completions, image_paths, normal_captions)):
#             # try:
#         # 1. 提取助手回复内容
#             if isinstance(completion, list) and completion and isinstance(completion[0], dict):
#                 content = completion[0].get("content", "")
#             else:
#                 content = str(completion)
#
#             # 2. 尝试从内容中提取<answer>标签
#             print("content:", content)
#             answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
#             if not answer_match:
#                 logger.warning(f"No <answer> tag found in sample {i}")
#                 rewards.append(0.0)
#                 continue
#
#             answer_content = answer_match.group(1).strip()
#
#             # 3. 尝试解析边界框
#             bbox = None
#
#             # 尝试1: 解析JSON格式
#             try:
#                 json_match = re.search(r'\{.*\}', answer_content, re.DOTALL)
#                 if json_match:
#                     json_str = json_match.group(0)
#                     bbox_data = json.loads(json_str)
#
#                     if isinstance(bbox_data, dict) and "bbox_2d" in bbox_data:
#                         bbox = bbox_data["bbox_2d"]
#                     elif isinstance(bbox_data, list) and bbox_data and isinstance(bbox_data[0], dict):
#                         if "bbox_2d" in bbox_data[0]:
#                             bbox = bbox_data[0]["bbox_2d"]
#             except:
#                 pass
#
#             # 尝试2: 解析数组格式
#             if bbox is None:
#                 list_match = re.search(r'$$(\d+),\s*(\d+),\s*(\d+),\s*(\d+)$$', answer_content)
#                 if list_match:
#                     bbox = [int(list_match.group(1)), int(list_match.group(2)),
#                             int(list_match.group(3)), int(list_match.group(4))]
#
#             # 4. 验证边界框
#             if not bbox or len(bbox) != 4 or not all(isinstance(c, (int, float)) for c in bbox):
#                 logger.warning(f"Invalid bbox in sample {i}")
#                 rewards.append(0.0)
#                 continue
#
#             print("bbox:", bbox)
#             # 5. 处理图像
#             image = Image.open(image_path).convert("RGB")
#             width, height = image.size
#
#             # 确保边界框在图像范围内
#             x1, y1, x2, y2 = [int(c) for c in bbox]
#             x1 = max(0, min(x1, width - 1))
#             y1 = max(0, min(y1, height - 1))
#             x2 = max(0, min(x2, width - 1))
#             y2 = max(0, min(y2, height - 1))
#
#             if x1 >= x2 or y1 >= y2:
#                 logger.warning(f"Invalid bbox coordinates in sample {i}")
#                 rewards.append(0.0)
#                 continue
#
#             # 裁剪并调整到模型期望的大小 (使用元组格式)
#             cropped_img = image.crop((x1, y1, x2, y2)).resize(self.image_size)
#
#             # 6. 使用处理器准备输入
#             inputs = self.clip_processor(
#                 text=[caption],
#                 images=cropped_img,
#                 padding="max_length",
#                 return_tensors="pt"
#             ).to(self.device)
#
#             # 确保pixel_values维度正确
#             if inputs.pixel_values.dim() == 3:
#                 inputs.pixel_values = inputs.pixel_values.unsqueeze(0)
#
#             # 核心修改：复制输入使batch size=2 (从[1,3,256,256] -> [2,3,256,256])
#             for key in inputs:
#                 if torch.is_tensor(inputs[key]):
#                     # 计算复制参数：batch维度复制2次，其他维度不变
#                     repeat_times = [2] + [1] * (inputs[key].dim() - 1)
#                     inputs[key] = inputs[key].repeat(*repeat_times)
#
#             # 7. 模型推理
#             print('inputs:', inputs.pixel_values.shape)
#             print('embeddings_1:', self.conv.weight.shape)
#             print('embeddings_2:', self.clip_model.vision_model.embeddings.patch_embedding.weight.shape)
#             # pixel_values = pixel_values.cuda
#             # conv_out = self.conv(inputs.pixel_values.cpu())
#             # print('conv_out:', conv_out.shape)
#             outputs = self.clip_model(**inputs)
#
#             # 8. 处理输出：提取logits
#             if hasattr(outputs, 'logits_per_image'):
#                 logits = outputs.logits_per_image
#             elif hasattr(outputs, 'logits'):
#                 logits = outputs.logits
#             else:
#                 if isinstance(outputs, tuple) and len(outputs) > 0:
#                     logits = outputs[0]
#                 else:
#                     logits = outputs
#
#             # 计算概率时只取第一个样本的结果
#             probs = torch.sigmoid(logits)
#
#             # 策略：根据维度取第一个样本的分数
#             if probs.dim() == 1:  # 一维张量 [2] -> 取索引0
#                 reward = probs[0].item()
#             elif probs.dim() == 2:  # 二维张量 [2,1] -> 取[0,0]
#                 reward = probs[0, 0].item()
#             else:  # 高维张量展平后取第一个元素
#                 reward = probs.reshape(-1)[0].item()
#
#             rewards.append(reward)
#             processed += 1
#             logger.info(f"Sample {i} reward: {reward:.4f}")
#
#             # except Exception as e:
#             #     logger.error(f"Error processing sample {i}: {e}")
#             #     rewards.append(0.0)
#
#         logger.info(f"Successfully processed {processed}/{total_samples} samples")
#         return rewards


#
#
# import re
#
# import transformers
# from PIL import Image
# from torch import nn
# from torchvision.transforms import ToTensor
# from transformers import AutoModel, AutoProcessor
# import torch
#
# class CLIPFunction(nn.Module):
#     def __init__(self, model_name_or_path=None):
#         super().__init__()
#         self.model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP", cache_dir="/mnt/dataset/zhangjiafan/huggingface")
#         self.processor = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP", cache_dir="/mnt/dataset/zhangjiafan/huggingface")
#
#
#     @torch.no_grad()
#     def forward(self,
#                 completions,
#                 image_paths=None,
#                 normal_captions=None,
#                ):
#         device = self.model.device
#         self.model.eval()
#
#         rewards = []
#         answer_tag_pattern = r'<answer>(.*?)</answer>'
#         bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
#         print("completions:", completions)
#         for completion, image_path, normal_caption in zip(completions, image_paths, normal_captions):
#             content = completion[0]["content"]
#             print("content:", content)
#             content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
#             if content_answer_match:
#                 content_answer = content_answer_match.group(1).strip()
#                 bbox_match = re.search(bbox_pattern, content_answer)
#                 if bbox_match:
#                     bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)),
#                             int(bbox_match.group(4))]
#                     # try:
#                     image = Image.open(image_path)
#                     print('image:',image)
#                     cropped_image = image.crop(bbox).resize((256, 256))
#                     print('cropped_image:',cropped_image)
#                     text = [normal_caption]  # 使用数据集中的 normal_caption 作为匹配文本描述
#                     print('text:',text)
#                     inputs = self.processor(text=text, images=cropped_image, padding="max_length",
#                                             return_tensors="pt").to(device)
#                     print('inputs_input_ids:', inputs['input_ids'].size())
#                     print('inputs_pixel_values:',inputs['pixel_values'].size())
#                     outputs = self.model(**inputs)
#                     print(outputs.shape)
#                     logits_per_image = outputs.logits_per_image
#                     probs = torch.sigmoid(logits_per_image)
#                     reward = probs.item()
#                     # except Exception as e:
#                     #     print(f"Error computing semantic consistency reward: {e}")
#                     #     reward = 0.0
#                 else:
#                     reward = 0.0
#             else:
#                 reward = 0.0
#             rewards.append(reward)
#         return rewards

#
# import re
# import transformers
# from PIL import Image
# from torch import nn
# from transformers import AutoModel, AutoProcessor
# import torch
#
# class RSCLIPmodel(nn.Module):
#     def __init__(self):
#         super(RSCLIPmodel,self).__init__()
#         self.model = AutoModel.from_pretrained(
#             "lcybuaa/Git-RSCLIP-base",
#             cache_dir="/mnt/dataset/zhangjiafan/huggingface",
#             torch_dtype=torch.float32
#         )
#     def forward(self, input):
#         return self.model(**input)
#
#
# class CLIPFunction(nn.Module):
#     def __init__(self, model_name_or_path=None):
#         super().__init__()
#         # 不使用 device_map 和 low_cpu_mem_usage 参数
#         self.conv = nn.Conv2d(3,1024,16,16)
#         # self.model = AutoModel.from_pretrained(
#         #     "lcybuaa/Git-RSCLIP-base",
#         #     cache_dir="/mnt/dataset/zhangjiafan/huggingface",
#         #     torch_dtype=torch.float32
#         # )
#         # 显式将模型移动到 CPU
#         self.model = RSCLIPmodel()
#         self.model = self.model.cpu()
#         self.model.eval()
#
#         self.processor = AutoProcessor.from_pretrained(
#             "lcybuaa/Git-RSCLIP-base",
#             cache_dir="/mnt/dataset/zhangjiafan/huggingface"
#         )
#
#     @torch.no_grad()
#     def forward(self,
#                 completions,
#                 image_paths=None,
#                 normal_captions=None,
#                 ):
#
#         rewards = []
#         answer_tag_pattern = r'<answer>(.*?)</answer>'
#         bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
#
#         for completion, image_path, normal_caption in zip(completions, image_paths, normal_captions):
#             content = completion[0]["content"]
#             content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
#
#             print("content:", content)
#             print("image_path:", image_path)
#             print("normal_caption:", normal_caption)
#
#             if content_answer_match:
#                 content_answer = content_answer_match.group(1).strip()
#                 bbox_match = re.search(bbox_pattern, content_answer)
#
#                 if bbox_match:
#                     bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)),
#                             int(bbox_match.group(3)), int(bbox_match.group(4))]
#
#                     # try:
#                     image = Image.open(image_path)
#                     cropped_image = image.crop(bbox).resize((256, 256))
#                     text = [normal_caption]
#
#                     # 在CPU上处理输入
#                     inputs = self.processor(
#                         text=text,
#                         images=cropped_image,
#                         padding="max_length",
#                         return_tensors="pt"
#                     )
#
#                     # 确保所有输入都在CPU上
#                     for key in inputs:
#                         inputs[key] = inputs[key].cpu()
#
#                     # 在CPU上运行模型
#                     outputs = self.model(inputs)
#                     logits_per_image = outputs.logits_per_image
#                     probs = torch.sigmoid(logits_per_image)
#                     reward = probs.item()
#
#                     # except Exception as e:
#                     #     print(f"Error computing semantic consistency reward: {e}")
#                     #     reward = 0.0
#                 else:
#                     reward = 0.0
#             else:
#                 reward = 0.0
#
#             rewards.append(reward)
#
#         return rewards
#


#
# clip_client.py
import re
import base64
from PIL import Image
from io import BytesIO
import requests
from torch import nn


class CLIPFunction(nn.Module):
    def __init__(self, url="http://172.169.0.2:5000"):
        super().__init__()  # 必须添加父类初始化
        self.url = f"{url}/score"
        self.answer_tag_pattern = r'<answer>(.*?)</answer>'
        self.bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'

    def _preprocess(self, completions, image_paths, normal_captions):
        processed = []
        for comp, img_path, cap in zip(completions, image_paths, normal_captions):
            try:
                content = comp[0]["content"]

                # 解析答案和边界框
                answer_match = re.search(self.answer_tag_pattern, content, re.DOTALL)
                if not answer_match:
                    continue

                bbox_match = re.search(self.bbox_pattern, answer_match.group(1))
                if not bbox_match:
                    continue

                bbox = list(map(int, bbox_match.groups()))

                # 处理图像
                img = Image.open(img_path)
                cropped_img = img.crop(bbox).resize((256, 256))

                # 转换为Base64
                buffered = BytesIO()
                cropped_img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                processed.append({
                    'image_base64': img_base64,
                    'normal_caption': cap
                })
            except Exception as e:
                print(f"预处理错误: {e}")
                continue

        return processed

    def __call__(self, completions, image_paths, normal_captions):
        inputs = self._preprocess(completions, image_paths, normal_captions)

        if not inputs:
            return [0.0] * len(completions)

        try:
            print('xxxx')
            response = requests.post(
                self.url,
                json={'inputs': inputs},
                timeout=30
            )
            print(response.json())
            return response.json()['rewards']
        except Exception as e:
            print(f"API请求错误: {e}")
            return [0.0] * len(completions)