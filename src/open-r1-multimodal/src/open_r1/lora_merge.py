import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载原始模型
model_name_or_path = "/mnt/dataset/zhangjiafan/huggingface/SegZero_model_cache/Qwen/Qwen2.5-VL-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)

# 加载LoRA模型
lora_model_path = "/mnt/dataset/zhangjiafan/Checkpoints/lora8_checkpoint-12180"
model = PeftModel.from_pretrained(model, lora_model_path)

# 合并LoRA权重到基础模型
model = model.merge_and_unload()

# 保存合并后的模型
output_dir = "/mnt/dataset/zhangjiafan/Checkpoints/merge_lora8"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)