from flask import Flask, request, jsonify
import torch
import base64
from PIL import Image
from io import BytesIO
from transformers import AutoModel, AutoProcessor

app = Flask(__name__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 初始化模型
model = AutoModel.from_pretrained("lcybuaa/Git-RSCLIP", cache_dir="/mnt/dataset/zhangjiafan/huggingface")
processor = AutoProcessor.from_pretrained("lcybuaa/Git-RSCLIP", cache_dir="/mnt/dataset/zhangjiafan/huggingface")
model = model.to(device)  # 确保模型移动到正确设备
model.eval()
print("模型加载完成")


@app.route('/score', methods=['POST'])
def score():
    try:
        data = request.json
        inputs = data.get('inputs', [])
        rewards = []

        for item in inputs:
            try:
                # 解码Base64图像
                img_data = base64.b64decode(item['image_base64'])
                cropped_img = Image.open(BytesIO(img_data))
                normal_caption = item['normal_caption']

                # 模型推理
                with torch.no_grad():
                    inputs = processor(
                        text=[normal_caption],
                        images=cropped_img,
                        padding=True,
                        return_tensors="pt"
                    ).to(device)

                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = torch.sigmoid(logits_per_image)
                    rewards.append(probs.item())

            except Exception as e:
                print(f"处理错误: {e}")
                rewards.append(0.0)  # 出错时添加默认值

        return jsonify({'rewards': rewards})

    except Exception as e:
        print(f"服务器错误: {e}")
        return jsonify({'rewards': [0.0] * len(inputs)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 启用调试模式