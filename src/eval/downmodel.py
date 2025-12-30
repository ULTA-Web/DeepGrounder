# from transformers import AutoProcessor, AutoModelForImageTextToText
#
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
# model = AutoModelForImageTextToText.from_pretrained("omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps", device_map="auto",cache_dir="E:/model_cache")

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
#
# # Load the model in half-precision on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained("omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps", device_map="auto",cache_dir="E:/model_cache")
