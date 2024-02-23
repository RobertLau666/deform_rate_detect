import os
import argparse
import json
from PIL import Image
import torch.nn as nn
from transformers import BeitFeatureExtractor, BeitForImageClassification, AutoImageProcessor, ConvNextV2ForImageClassification, ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import pandas as pd
import wandb
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import onnxruntime as ort
import random
from yolo_ import _image_preprocess, _data_postprocess
import time
from torchvision import transforms
import math
from typing import List

def parse():
    parser = argparse.ArgumentParser(description='Description of auto labelling and scoring')
    
    parser.add_argument('--image_dir_file', type=str, default='test_images/', help='test image directory')
    parser.add_argument('--batch_size', type=int, default=40, help='detects the batch size of the data.')
    parser.add_argument('--max_infer_size', type=int, default=640, help='xx.')
    parser.add_argument('--conf_threshold', type=float, default=0.3, help='xx.')
    parser.add_argument('--iou_threshold', type=float, default=0.7, help='xx.')
    parser.add_argument('--_LABELS', type=str, default=['head', 'face'], help='xx.', nargs='+')
    parser.add_argument('--default_prob', type=float, default=0.5, help='If no detection method has been added, set a prob for now.')
    parser.add_argument('--score_save_dir', type=str, default='deform_rate_score/', help='the directory where the score json is saved.')
    
    args = parser.parse_args()
    return args

class DetectModel:
    def __init__(self):
        self.head_det_session = ort.InferenceSession("/dfs/comicai/zhiyuan.shi/models/deepghs/head_detect/head_detect_best_s.onnx", providers=['CUDAExecutionProvider'])

        self.processor_face = ViTImageProcessor.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/vit-base-patch16-224_face/output_face/checkpoint-730')
        self.model_face = ViTForImageClassification.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/vit-base-patch16-224_face/output_face/checkpoint-730').to(device)
        self.model_face.eval()
        
        self.feature_extractor_lowerbody = BeitFeatureExtractor.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_lower_deit/log_outdir/checkpoint-1424')
        self.model_lowerbody = BeitForImageClassification.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_lower_deit/log_outdir/checkpoint-1424').to(device)
        self.model_lowerbody.eval()
        
        self.preprocessor_upperbody = AutoImageProcessor.from_pretrained("/dfs/comicai/tong.liu/code/image classifier/deformity_upper_convnetv2/outdir/checkpoint-1458")
        self.model_upperbody = ConvNextV2ForImageClassification.from_pretrained("/dfs/comicai/tong.liu/code/image classifier/deformity_upper_convnetv2/outdir/checkpoint-1458").to(device)
        self.model_upperbody.eval()
        
        self.feature_extractor_action = BeitFeatureExtractor.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_action_deit/outdir_action/checkpoint-5405')
        self.model_action = BeitForImageClassification.from_pretrained('/dfs/comicai/tong.liu/code/image classifier/deformity_action_deit/outdir_action/checkpoint-5405').to(device)
        self.model_action.eval()

    def calculate_face_deformation(self, batch_images_face):
        # 1.1 nose_deformation (76/183)
        # 1.2 mouth_deformation (26/183)
        # 1.3 eyebrow_deformation (130/183)
        # 1.4 proportional_errors (0/183)
        # 1.5 asymmetrical_features (4/183)
        # 1.6 inconsistent_skin_tone (0/183)
        inputs_face = self.processor_face(images=batch_images_face, return_tensors="pt").to(device)
        outputs_face = self.model_face(**inputs_face)
        logits_face = outputs_face.logits.cpu().numpy()
        probs_face = torch.nn.functional.softmax(torch.tensor(logits_face), dim=-1)
        probs_face_deformation = [probs_face_[0].item() for probs_face_ in probs_face]

        return probs_face_deformation

    def calculate_hair_deformation(self, image_paths):
        # 2.1 unnaturally_hair_strands (163/183)
        # 2.2 inconsistent_hair_color (37/183)
        # 2.3 hair_merging (134/183)

        probs_hair_deformation = [args.default_prob] * len(image_paths)

        return probs_hair_deformation

    def calculate_eyes_deformation(self, image_paths):
        # 3.1 skewed_iris_and_pupils (163/183)
        # 3.2 asymmetrical_eyes (146/183)
        # 3.3 misproportion (9/183)
        # 3.4 inappropriate_eyelids (122/183)

        probs_eyes_deformation = [args.default_prob] * len(image_paths)

        return probs_eyes_deformation

    def calculate_ear_deformation(self, image_paths):
        # 4.1 placement_errors (13/183)
        # 4.2 lack_of_antihelix_detail (90/183)
        # 4.3 inconsistent_styles (1/183)
        # 4.4 proportion_problems (2/183)

        probs_ear_deformation = [args.default_prob] * len(image_paths)

        return probs_ear_deformation

    def calculate_upperbody_deformation(self, batch_images):
        # 5.1 Upperbody deformation (21/183)
        #   5.1.1 inconsistent_anatomy (38/183)
        #   5.1.2 inconsistent_skin_color (0/183)
        #   5.1.3 body_merging (3/183)
        inputs_upperbody = self.preprocessor_upperbody(images=batch_images, return_tensors="pt").to(device)
        outputs_upperbody = self.model_upperbody(**inputs_upperbody)
        logits_upperbody = outputs_upperbody.logits.cpu().numpy()
        probs_upperbody = torch.nn.functional.softmax(torch.tensor(logits_upperbody), dim=-1)
        probs_upperbody_deformation = [probs_upper_[0].item() for probs_upper_ in probs_upperbody]

        return probs_upperbody_deformation

    def calculate_lowerbody_deformation(self, batch_images):
        # 5.2 lowerbody deformation (17/183)
        #   5.2.1 inconsistent_anatomy (38/183)
        #   5.2.2 inconsistent_skin_color (0/183)
        #   5.2.3 body_merging (3/183)
        inputs_lowerbody = self.feature_extractor_lowerbody(images=batch_images, return_tensors="pt").to(device)
        outputs_lowerbody = self.model_lowerbody(**inputs_lowerbody)
        logits_lowerbody = outputs_lowerbody.logits.cpu().numpy()
        probs_lowerbody = torch.nn.functional.softmax(torch.tensor(logits_lowerbody), dim=-1)
        probs_lowerbody_deformation = [probs_lowerbody_[0].item() for probs_lowerbody_ in probs_lowerbody]

        return probs_lowerbody_deformation

    def calculate_hand_deformation(self, batch_images):
        # 5.3 Hand deformation (106/183)
        #   57%
        #   5.3.1 deformed_hand (106/183)
        #   5.3.2 no_hand_exist (65/183)
        probs_hand_deformation = [args.default_prob] * len(batch_images)

        return probs_hand_deformation

    def calculate_action_deformation(self, batch_images):
        # deformed_action (7/183)
        # no_action_exist (11/183)
        inputs_action = self.feature_extractor_action(images=batch_images, return_tensors="pt").to(device)
        outputs_action = self.model_action(**inputs_action)
        logits_action = outputs_action.logits.cpu().numpy()
        probs_action = torch.nn.functional.softmax(torch.tensor(logits_action), dim=-1)
        probs_action_deformation = [probs_action_[0].item() for probs_action_ in probs_action]

        return probs_action_deformation

    def calculate_clothing_and_accessories(self, image_paths):
        # good_cloth_accessory (38/183)
        # bad_cloth_accessory (139/183)

        probs_clothing_and_accessories = [args.default_prob] * len(image_paths)

        return probs_clothing_and_accessories

    def calculate_watermarks(self, image_paths):
        # no_watermarks
        # have_watermarks
        probs_watermarks = [args.default_prob] * len(image_paths)

        return probs_watermarks

    def calculate_black_white_borders(self, image_paths):
        # no_borders_issues
        # have_borders_issues

        probs_black_white_borders = [args.default_prob] * len(image_paths)

        return probs_black_white_borders

    def calculate_solid_color_background(self, image_paths):
        # no_solid_color_background
        # have_solid_color_background (6/183)
        probs_solid_color_background = [args.default_prob] * len(image_paths)

        return probs_solid_color_background

    def calculate_global_image_deformation(self, image_paths):
        # This is a deformity score derived from the global level of the picture.
        probs_global_image_deformation = [args.default_prob] * len(image_paths)

        return probs_global_image_deformation
    
    def model_inference(self, image_paths, batch_images, batch_images_face):
        with torch.no_grad():
            probs_face_deformation = self.calculate_face_deformation(batch_images_face)
            probs_hair_deformation = self.calculate_hair_deformation(image_paths)
            probs_eyes_deformation = self.calculate_eyes_deformation(image_paths)
            probs_ear_deformation = self.calculate_ear_deformation(image_paths)
            probs_upperbody_deformation = self.calculate_upperbody_deformation(batch_images)
            probs_lowerbody_deformation = self.calculate_lowerbody_deformation(batch_images)
            probs_hand_deformation = self.calculate_hand_deformation(batch_images)
            probs_action_deformation = self.calculate_action_deformation(batch_images)
            probs_clothing_and_accessories = self.calculate_clothing_and_accessories(image_paths)
            probs_watermarks = self.calculate_watermarks(image_paths)
            probs_black_white_borders = self.calculate_black_white_borders(image_paths)
            probs_solid_color_background = self.calculate_solid_color_background(image_paths)
            probs_global_image_deformation = self.calculate_global_image_deformation(image_paths)
        
        probs_dict = {
            "probs_face_deformation": probs_face_deformation,
            "probs_hair_deformation": probs_hair_deformation,
            "probs_eyes_deformation": probs_eyes_deformation,
            "probs_ear_deformation": probs_ear_deformation,
            "probs_upperbody_deformation": probs_upperbody_deformation,
            "probs_lowerbody_deformation": probs_lowerbody_deformation,
            "probs_hand_deformation": probs_hand_deformation,
            "probs_action_deformation": probs_action_deformation,
            "probs_clothing_and_accessories": probs_clothing_and_accessories,
            "probs_watermarks": probs_watermarks,
            "probs_black_white_borders": probs_black_white_borders,
            "probs_solid_color_background": probs_solid_color_background,
            "probs_global_image_deformation": probs_global_image_deformation
        }

        for key, value in probs_dict.items():
            print(f"{key}: {value}")

        return probs_dict


def data_prepare(image_paths, start_idx, end_idx):
    batch_images = []
    batch_images_face = []

    for idx in range(start_idx, end_idx):
        image_path = image_paths[idx]
        image = Image.open(image_path)
        batch_images.append(image)

        new_image, old_size, new_size = _image_preprocess(image, args.max_infer_size)
        numpy_image = np.array(new_image).astype(np.float32)
        numpy_image = numpy_image / 255.0
        numpy_image = np.transpose(numpy_image, (2, 0, 1))
        numpy_image = np.expand_dims(numpy_image, axis=0)
        output, = detect_model.head_det_session.run(['output0'], {'images': numpy_image})
        boxes = _data_postprocess(output[0], args.conf_threshold, args.iou_threshold, old_size, new_size, args._LABELS)
        if len(boxes) == 0:
            # Crop randomly a 224x224 image.
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

    return batch_images, batch_images_face


def record_json(image_names, probs_dict):
    for i, image_name in enumerate(image_names):
        score_dict = {}
        for key, value in probs_dict.items():
            score_dict[key] = value[i]
        
        final_score = 0
        weight_sum = 0
        for key, value in weight_dict.items():
            final_score += score_dict["probs_" + key] * weight_dict[key]
            weight_sum += weight_dict[key]
        score_dict["final_score"] = final_score / weight_sum

        if not os.path.exists(args.score_save_dir):
            os.makedirs(args.score_save_dir, exist_ok=True)
        with open(os.path.join(args.score_save_dir, image_name + '.json'), "w") as file:
            file.write(json.dumps(score_dict, indent=4))


def calculate_score(image_names, image_paths):
    num_images = len(image_paths)
    num_batches = (num_images + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx}/{num_batches}...")
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, num_images)

        # prepare data
        batch_images, batch_images_face = data_prepare(image_paths, start_idx, end_idx)
        # model inference
        probs_dict = detect_model.model_inference(image_paths, batch_images, batch_images_face)
        # record as json
        record_json(image_names, probs_dict)


def get_image_names_paths(image_dir_file):
    image_paths = []

    if os.path.isdir(image_dir_file):
        image_names = os.listdir(image_dir_file)
        image_names = [image_name for image_name in image_names if image_name not in ['.ipynb_checkpoints']]
        for image_name in image_names:
            image_path = os.path.join(image_dir_file, image_name)
            image_paths.append(image_path)
    elif os.path.isfile(image_dir_file):
        file_name, file_extension = os.path.splitext(image_dir_file)
        if file_extension == 'txt':
            with open(image_dir_file, 'r') as file:
                for line in file:
                    line = line.strip()
                    image_paths.append(line)
        elif file_extension in ['jpeg', 'jpg', 'png', 'webp']:
            image_paths.append(image_dir_file)
    image_names = sorted([os.path.splitext(os.path.basename(image_path))[0] for image_path in image_paths])
    image_paths = sorted(image_paths)

    return image_names, image_paths


def main():
    image_names, image_paths = get_image_names_paths(args.image_dir_file)
    calculate_score(image_names, image_paths)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dict = {
        "face_deformation": 0.5,
        "hair_deformation": 0.5,
        "eyes_deformation": 0.5,
        "ear_deformation": 0.5,
        "upperbody_deformation": 0.5,
        "lowerbody_deformation": 0.5,
        "hand_deformation": 0.5,
        "action_deformation": 0.5,
        "clothing_and_accessories": 0.5,
        "watermarks": 0.5,
        "black_white_borders": 0.5,
        "solid_color_background": 0.5,
        "global_image_deformation": 0.9
    }
    deformation_types = weight_dict.keys()

    detect_model = DetectModel()
    args = parse()

    main()
