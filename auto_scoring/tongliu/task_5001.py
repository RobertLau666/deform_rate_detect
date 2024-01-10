import torch.nn as nn
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
import os
import pandas as pd
import wandb
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from transformers import BeitFeatureExtractor, BeitForImageClassification
from PIL import Image
import requests
import pandas as pd
import wandb
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import numpy as np

from datasets import load_dataset

from transformers import logging
import torch.nn.functional as F
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification
import torch
from datasets import load_dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
import onnxruntime as ort
import numpy as np
import random

from yolo_ import _image_preprocess, _data_postprocess
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

import time
import torch
from torchvision import transforms



import math
from typing import List

import numpy as np
from PIL import Image


class InferenceModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.head_det_session = ort.InferenceSession("/dfs/comicai/zhiyuan.shi/models/deepghs/head_detect/head_detect_best_s.onnx", providers=['CUDAExecutionProvider'])
        self.max_infer_size = 640
        self._LABELS = ['head']
        self.conf_threshold = 0.3
        self.iou_threshold = 0.7
        
        self.processor_face = ViTImageProcessor.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/vit-base-patch16-224_face/output_face/checkpoint-730')
        self.model_face = ViTForImageClassification.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/vit-base-patch16-224_face/output_face/checkpoint-730').to(self.device)
        self.model_face.eval()
        
        self.feature_extractor_lowerbody = BeitFeatureExtractor.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_lower_deit/log_outdir/checkpoint-1424')
        self.model_lowerbody = BeitForImageClassification.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_lower_deit/log_outdir/checkpoint-1424').to(self.device)
        self.model_lowerbody.eval()
        
        self.preprocessor_upperbody = AutoImageProcessor.from_pretrained("/dfs/comicai/tong.liu/code/image classifier/deformity_upper_convnetv2/outdir/checkpoint-1458")
        self.model_upperbody = ConvNextV2ForImageClassification.from_pretrained("/dfs/comicai/tong.liu/code/image classifier/deformity_upper_convnetv2/outdir/checkpoint-1458").to(self.device)
        self.model_upperbody.eval()
        
        self.feature_extractor_action = BeitFeatureExtractor.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_action_deit/outdir_action/checkpoint-5405')
        self.model_action = BeitForImageClassification.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_action_deit/outdir_action/checkpoint-5405').to(self.device)
        self.model_action.eval()
        self.image_savedir = "./0104score.txt"
    
    
    def inference(self, image_paths, batch_size=40):
        num_images = len(image_paths)
        num_batches = (num_images + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            result_dict = {}

            print(f"Processing batch {batch_idx}.")
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_images)
            batch_images = []
            batch_images_face = []
            
            for idx in range(start_idx, end_idx):
                image_path = image_paths[idx]
                image = Image.open(image_path)
                batch_images.append(image)

                new_image, old_size, new_size = _image_preprocess(image, self.max_infer_size)
                numpy_image = np.array(new_image).astype(np.float32)
                numpy_image = numpy_image / 255.0
                numpy_image = np.transpose(numpy_image, (2, 0, 1))
                numpy_image = np.expand_dims(numpy_image, axis=0)
                output, = self.head_det_session.run(['output0'], {'images': numpy_image})
                boxes = _data_postprocess(output[0], self.conf_threshold, self.iou_threshold, old_size, new_size, self._LABELS)
                if len(boxes) == 0:
                    # 随机截取一个 224x224 的图像
                    width, height = image.size
                    x1 = random.randint(0, width - 224)
                    y1 = random.randint(0, height - 224)
                    x2 = x1 + 224
                    y2 = y1 + 224
                    face_image = image.crop((x1, y1, x2, y2))
                    batch_images_face.append(face_image)
                else:
                    (x1, y1, x2, y2), _, _ = boxes[0]
                    face_image = image.crop((x1, y1, x2, y2))
                    batch_images_face.append(face_image)
                        
                # 省略了对图像的预处理和人脸检测部分
                
            with torch.no_grad():
                inputs_lowerbody = self.feature_extractor_lowerbody(images=batch_images, return_tensors="pt").to(self.device)
                outputs_lowerbody = self.model_lowerbody(**inputs_lowerbody)
                logits_lowerbody = outputs_lowerbody.logits.cpu().numpy()
                probs_lowerbody = torch.nn.functional.softmax(torch.tensor(logits_lowerbody), dim=-1)

                inputs_face = self.processor_face(images=batch_images_face, return_tensors="pt").to(self.device)
                outputs_face = self.model_face(**inputs_face)
                logits_face = outputs_face.logits.cpu().numpy()
                probs_face = torch.nn.functional.softmax(torch.tensor(logits_face), dim=-1)

                inputs_upper = self.preprocessor_upperbody(images=batch_images, return_tensors="pt").to(self.device)
                outputs_upper = self.model_upperbody(**inputs_upper)
                logits_upper = outputs_upper.logits.cpu().numpy()
                probs_upper = torch.nn.functional.softmax(torch.tensor(logits_upper), dim=-1)

                inputs_action = self.feature_extractor_action(images=batch_images, return_tensors="pt").to(self.device)
                outputs_action = self.model_action(**inputs_action)
                logits_action = outputs_action.logits.cpu().numpy()
                probs_action = torch.nn.functional.softmax(torch.tensor(logits_action), dim=-1)
                        # 使用模型进行推理
                # 省略了模型推理部分
                
            for idx in range(start_idx, end_idx):
                predicted_prob_lowerbody = probs_lowerbody[idx - start_idx][1].item()
                lowerbody = "normal lowerbody" if predicted_prob_lowerbody >= 0.5 else "abnormal lowerbody"
                score_lowerbody = predicted_prob_lowerbody

                predicted_prob_face = probs_face[idx - start_idx][1].item()
                face = "normal face" if predicted_prob_face >= 0.5 else "abnormal face"
                score_face = predicted_prob_face

                predicted_prob_upperbody = probs_upper[idx - start_idx][1].item()
                upperbody = "normal upperbody" if predicted_prob_upperbody >= 0.5 else "abnormal upperbody"
                score_upperbody = predicted_prob_upperbody


                predicted_prob_action = probs_action[idx - start_idx][1].item()
                action = "normal action" if predicted_prob_action >= 0.5 else "abnormal action"
                score_action = predicted_prob_action


                # 需要保存图片的话就是这里。
                # number = 0.8
                # if (score_lowerbody >= number) and (score_face >= number) and (score_upperbody >= number) and (score_action >= number):
                #     image_save = batch_images[idx - start_idx]
                #     image_save.save(f"/dfs/comicai/tong.liu/code/inference/datatime/image/image1220_0.8/photo_{batch_idx}_{idx}_L_{score_lowerbody}_F_{score_face}_U_{score_upperbody}_A_{score_action}.png")
                

                # number1 = 0.7
                # number2 = 0.8
                # if (score_lowerbody >= number1) and (score_face >= number) and (score_upperbody >= number) and (score_action >= number):
                #     image_save = batch_images[idx - start_idx]
                #     image_save.save(f"/dfs/comicai/tong.liu/code/inference/datatime/image/image1219_0.8/photo_{batch_idx}_{idx}_L_{score_lowerbody}_F_{score_face}_U_{score_upperbody}_A_{score_action}.png")





                # 处理推理结果
                # 省略了推理结果处理部分
                
                # 保存图片的部分也可以在这里进行
                
                result_dict[image_paths[idx]] = {
                    "score_lowerbody": score_lowerbody,
                    "score_face": score_face,
                    "score_upperbody": score_upperbody,
                    "score_action": score_action
                }
            with open(self.image_savedir, "a") as file:
                # 写入推理结果到文件
                for key, value in result_dict.items():
                    file.write(f"{key}: {value}\n")
                print("Length:", len(result_dict))




image_paths = []
# result_dict = {}
# /dfs/comicai/tong.liu/code/inference/niji_total.txt      
with open('./2023_12_20.txt', 'r') as file:
    for line in file:
        line = line.strip()
        image_paths.append(line)


# 创建类实例并进行推理
model = InferenceModel()
# image_paths = [...]  # 读取图像路径列表
model.inference(image_paths)
