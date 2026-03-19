import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

import os
import json
import time
from tqdm import tqdm
import argparse

from PIL import Image
from groundingdino.util.inference import Model
from transformers import CLIPProcessor, CLIPModel

# Import COCO official evaluation tools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile  # For temporary file storage of COCO format data


# -------------------------- Global Configuration and Model Loading --------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# GroundingDINO configuration
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# CLIP configuration
CLIP_MODEL_PATH = "./pretrained/clip/"

# Load models
grounding_dino_model = Model(
    model_config_path=GROUNDING_DINO_CONFIG_PATH,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH
)

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE).eval()
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)

# ImageNet templates (for Local CLIP)
imagenet_templates = [
    'a photo of a {}.', 'a clear photo of the {}.', 'a photo of one {}.',
    'a close-up photo of the {}.', 'a bright photo of the {}.', 'a photo of the {}.',
    'a good photo of the {}.', 'a photo of a nice {}.', 'a photo of the cool {}.'
]


# -------------------------- Core Utility Functions --------------------------
def load_image(image_path):
    """Load and preprocess image (adapted for GroundingDINO input)"""
    image_pil = Image.open(image_path).convert("RGB")
    image_pil = image_pil.resize((512, 512))  # Uniform scaling to 512x512
    return np.array(image_pil), image_pil


def calc_clip_score(image_pil, prompt, use_template=False):
    """Calculate CLIP score (works for both global and local evaluations)"""
    prompts = [t.format(prompt) for t in imagenet_templates] if use_template else [prompt]
    with torch.no_grad():
        inputs = clip_processor(text=prompts, images=image_pil, return_tensors="pt", padding=True).to(DEVICE)
        outputs = clip_model(**inputs)
        return torch.mean(outputs.logits_per_image).cpu().item()


def compute_box_miou(gt_bbox, pred_bbox):
    """Calculate mIoU between two bounding boxes (in xyxy format)"""
    x1, y1 = max(gt_bbox[0], pred_bbox[0]), max(gt_bbox[1], pred_bbox[1])
    x2, y2 = min(gt_bbox[2], pred_bbox[2]), min(gt_bbox[3], pred_bbox[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    union = gt_area + pred_area - intersection
    return intersection / union


def detect_and_evaluate_instance(image_np, prompt, gt_bbox, miou_threshold=0.5):
    """Detect and evaluate a single instance (returns success flag, mIoU, prediction box, confidence)"""
    prompt = prompt.replace("a ", "")
    # GroundingDINO detection
    detections = grounding_dino_model.predict_with_classes(
        image=image_np, classes=[prompt], box_threshold=0.25, text_threshold=0.25
    )
    # NMS to filter duplicate boxes
    if len(detections.xyxy) > 0:
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), torch.from_numpy(detections.confidence), iou_threshold=0.8
        ).numpy().tolist()
        detections.xyxy, detections.confidence = detections.xyxy[nms_idx], detections.confidence[nms_idx]
    
    # Calculate maximum mIoU
    max_miou, best_pred_bbox = 0.0, None
    if len(detections.xyxy) > 0:
        for pred_bbox in detections.xyxy:
            miou = compute_box_miou(gt_bbox, pred_bbox)
            if miou > max_miou:
                max_miou, best_pred_bbox = miou, pred_bbox
    
    inst_success = 1 if max_miou >= miou_threshold else 0
    confidence = detections.confidence[0] if len(detections.confidence) > 0 else 0.0
    return inst_success, max_miou, best_pred_bbox, confidence


# -------------------------- Helper: Convert numpy types to native Python types --------------------------
def convert_numpy_types(obj):
    """Convert numpy data types to native Python types for proper JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# -------------------------- COCO Format Conversion + Official AP Calculation --------------------------

def get_coco_ap_metrics(gen_image_paths, coco_data, image_size=512):
    """AP、AP50、AP75"""
    coco_gt = {
        "info": {
            "description": "Temporary evaluation dataset",
            "url": "",
            "version": "1.0",
            "year": time.localtime().tm_year,
            "contributor": "",
            "date_created": time.strftime("%Y-%m-%d", time.localtime())
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown License",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "object", "supercategory": "object"}]
    }
    coco_pred = []

    img_id_map = {}  
    ann_id = 1      

    for img_path in gen_image_paths:
        img_name = os.path.basename(img_path)
        custom_img_id = img_name.split('_')[-1].split('.')[0]
        if custom_img_id not in coco_data:
            continue

        if custom_img_id not in img_id_map:
            coco_img_id = len(img_id_map) + 1
            img_id_map[custom_img_id] = coco_img_id
            coco_gt["images"].append({
                "id": coco_img_id,
                "file_name": img_name,
                "width": image_size,
                "height": image_size,
                "license": 1,
                "date_captured": ""
            })
        else:
            coco_img_id = img_id_map[custom_img_id]

        gt_info = coco_data[custom_img_id]
        gt_instances = gt_info["segment"]
        for inst in gt_instances:
            gt_bbox_xyxy = np.array(inst["bbox"]) * image_size
            x1, y1, x2, y2 = convert_numpy_types(gt_bbox_xyxy)
            w, h = x2 - x1, y2 - y1
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": coco_img_id,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "ignore": 0,
                "segmentation": [],
                "keypoints": []
            })
            ann_id += 1

        image_np, _ = load_image(img_path)
        for inst in gt_instances:
            inst_label = inst["label"].strip()
            gt_bbox_xyxy = np.array(inst["bbox"]) * image_size
            _, _, pred_bbox_xyxy, pred_conf = detect_and_evaluate_instance(
                image_np=image_np, prompt=inst_label, gt_bbox=gt_bbox_xyxy, miou_threshold=0.5
            )
            if pred_bbox_xyxy is None:
                continue
            
            x1, y1, x2, y2 = convert_numpy_types(pred_bbox_xyxy)
            w, h = x2 - x1, y2 - y1
            pred_conf = convert_numpy_types(pred_conf)
            
            coco_pred.append({
                "image_id": coco_img_id,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "score": pred_conf
            })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
        json.dump(coco_gt, gt_file, default=convert_numpy_types)
        gt_path = gt_file.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
        json.dump(coco_pred, pred_file, default=convert_numpy_types)
        pred_path = pred_file.name

    try:
        coco_gt_api = COCO(gt_path)
        coco_pred_api = coco_gt_api.loadRes(pred_path)
        coco_eval = COCOeval(coco_gt_api, coco_pred_api, iouType='bbox')
        coco_eval.params.catIds = [0]
        coco_eval.params.imgIds = coco_gt_api.getImgIds()
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap = round(coco_eval.stats[0], 4)
        ap50 = round(coco_eval.stats[1], 4)
        ap75 = round(coco_eval.stats[2], 4)

        return ap, ap50, ap75

    finally:
        os.unlink(gt_path)
        os.unlink(pred_path)
    


# -------------------------- Main Evaluation Logic --------------------------
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Evaluation with L2-L6 Level Statistics")
    parser.add_argument("--gen_dir", type=str, required=True, help="Directory with generated images (to evaluate)")
    parser.add_argument("--coco_path", type=str, required=True, help="Path to COCO format annotation file")
    parser.add_argument("--out_json", type=str, required=True, help="Path to COCO format annotation file")
    parser.add_argument("--miou_threshold", type=float, default=0.5, help="mIoU success threshold (default 0.5)")
    parser.add_argument("--metric_name", type=str, default="eval_result", help="Prefix for result file")
    parser.add_argument("--debug", action="store_true", help="Debug mode (print detailed information)")
    args = parser.parse_args()

    # -------------------------- 1. Load Data --------------------------
    gen_image_paths = [
        os.path.join(args.gen_dir, f)
        for f in os.listdir(args.gen_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    ]
    with open(args.coco_path, 'r') as f:
        coco_data = json.load(f)
    assert len(gen_image_paths) > 0, "Generated image directory is empty!"

    # -------------------------- 2. Initialize Metrics (L2-L6 Level Statistics) --------------------------
    level_stats = {
        'L2': {'total_images': 0, 'success_images': 0, 'total_instances': 0, 'success_instances': 0},
        'L3': {'total_images': 0, 'success_images': 0, 'total_instances': 0, 'success_instances': 0},
        'L4': {'total_images': 0, 'success_images': 0, 'total_instances': 0, 'success_instances': 0},
        'L5': {'total_images': 0, 'success_images': 0, 'total_instances': 0, 'success_instances': 0},
        'L6': {'total_images': 0, 'success_images': 0, 'total_instances': 0, 'success_instances': 0}
    }

    # Global metrics
    global_clip_total, local_clip_total = 0.0, 0.0
    clip_count, local_clip_count = 0, 0
    img_success_count, inst_success_count = 0, 0
    total_img, total_inst = 0, 0
    miou_total, miou_count = 0.0, 0

    # -------------------------- 3. Image-level Evaluation --------------------------
    for img_path in tqdm(gen_image_paths, desc="Evaluating Images"):
        img_name = os.path.basename(img_path)
        custom_img_id = img_name.split('_')[-1].split('.')[0]
        if custom_img_id not in coco_data:
            if args.debug:
                print(f"Skipping unannotated image: {img_name}")
            continue

        # Load GT information and image
        gt_info = coco_data[custom_img_id]
        caption, gt_instances = gt_info["caption"].strip(), gt_info["segment"]
        num_instances = len(gt_instances)
        
        # Determine image level (L2-L6)
        level = f'L{num_instances}' if 2 <= num_instances <= 6 else None
        
        total_img += 1
        image_np, image_pil = load_image(img_path)

        # Calculate global CLIP score
        global_clip = calc_clip_score(image_pil, caption, use_template=False)
        global_clip_total += global_clip
        clip_count += 1

        # Instance-level evaluation
        img_inst_success = 1  # Image-level success: all instances meet criteria
        
        for inst in gt_instances:
            total_inst += 1
            if level:
                level_stats[level]['total_instances'] += 1
                
            inst_label = inst["label"].strip()
            gt_bbox = np.array(inst["bbox"]) * 512  # Scale to 512x512

            # Instance detection and mIoU calculation
            inst_success, inst_miou, pred_bbox, _ = detect_and_evaluate_instance(
                image_np=image_np, prompt=inst_label, gt_bbox=gt_bbox, miou_threshold=args.miou_threshold
            )

            # Update instance-level metrics
            inst_success_count += inst_success
            if level and inst_success:
                level_stats[level]['success_instances'] += 1
                
            miou_total += inst_miou
            miou_count += 1
            img_inst_success &= inst_success

            # Calculate local CLIP score
            if pred_bbox is not None:
                x1, y1, x2, y2 = map(int, pred_bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(511, x2), min(511, y2)
                inst_crop = image_pil.crop((x1, y1, x2, y2)).resize((224, 224))
                local_clip = calc_clip_score(inst_crop, inst_label, use_template=True)
                local_clip_total += local_clip
                local_clip_count += 1

        # Update image-level success count
        img_success_count += img_inst_success
        
        # Update level-specific image statistics
        if level:
            level_stats[level]['total_images'] += 1
            if img_inst_success:
                level_stats[level]['success_images'] += 1

    # -------------------------- 4. Calculate Final Metrics --------------------------
    # Calculate success rates for each level
    level_metrics = {}
    for level in level_stats:
        stats = level_stats[level]
        success_rate = round(stats['success_images'] / stats['total_images'], 4) if stats['total_images'] > 0 else 0.0
        inst_success_rate = round(stats['success_instances'] / stats['total_instances'], 4) if stats['total_instances'] > 0 else 0.0
        
        level_metrics[f'{level}_success_rate'] = success_rate
        level_metrics[f'{level}_instance_success_rate'] = inst_success_rate

    # Global metrics
    global_clip_avg = round(global_clip_total / clip_count, 4) if clip_count > 0 else 0.0
    local_clip_avg = round(local_clip_total / local_clip_count, 4) if local_clip_count > 0 else 0.0
    img_success_rate = round(img_success_count / total_img, 4) if total_img > 0 else 0.0
    inst_success_rate = round(inst_success_count / total_inst, 4) if total_inst > 0 else 0.0
    miou_avg = round(miou_total / miou_count, 4) if miou_count > 0 else 0.0
    ap, ap50, ap75 = get_coco_ap_metrics(gen_image_paths, coco_data)

    # -------------------------- 5. Save and Print Results --------------------------
    metric_result = {
        "metric_name": args.metric_name,
        "gen_dir": args.gen_dir,
        "eval_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        # Image-Text Consistency
        "global_clip_avg": global_clip_avg,
        "local_clip_avg": local_clip_avg,
        # Global Spatial Accuracy
        "image_success_rate": img_success_rate,
        "instance_success_rate": inst_success_rate,
        "mIoU_avg": miou_avg,
        # COCO Official AP Metrics
        "AP": ap,
        "AP50": ap50,
        "AP75": ap75,
        # L2-L6 Level Metrics
        **level_metrics
    }

    # Save results to JSON file
    # output_path = f"./metric_{args.metric_name}.json"
    output_path = args.out_json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metric_result, f, indent=4, default=convert_numpy_types)  # Apply conversion function

    # Print final results
    print("\n" + "="*70)
    print("Final Evaluation Results (with L2-L6 Level Statistics)")
    print("="*70)
    print("Global Metrics:")
    print("-"*70)
    print(f"{'global_clip_avg':25}: {global_clip_avg}")
    print(f"{'local_clip_avg':25}: {local_clip_avg}")
    print(f"{'image_success_rate':25}: {img_success_rate}")
    print(f"{'instance_success_rate':25}: {inst_success_rate}")
    print(f"{'mIoU_avg':25}: {miou_avg}")
    print(f"{'AP':25}: {ap}")
    print(f"{'AP50':25}: {ap50}")
    print(f"{'AP75':25}: {ap75}\n")
    
    print("Level Metrics (L2-L6):")
    print("-"*70)
    for level in ['L2', 'L3', 'L4', 'L5', 'L6']:
        print(f"{level}:")
        print(f"  {'success_rate':23}: {level_metrics[f'{level}_success_rate']}")
        print(f"  {'instance_success_rate':23}: {level_metrics[f'{level}_instance_success_rate']}\n")
    
    print("="*70)
    print(f"Results saved to: {output_path}")
    