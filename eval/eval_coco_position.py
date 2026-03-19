import torch
import random
import argparse
import numpy as np
import json
import os
from PIL import Image
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont

from consist_compose import ConsistComposeBagelModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_simple_prompt(data_item: Dict[str, Any]) -> str:
    return data_item["ground_generate_prompt"]

def parse_bbox_info(bbox_info: list) -> list:
    parsed_bboxes = []
    for bbox in bbox_info:
        parsed_bboxes.append(bbox)
    return parsed_bboxes

def draw_bboxes_on_image(image: Image.Image, bboxes: list) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    border_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255)
    ]
    label_bg_color = (0, 0, 0)
    label_text_color = (255, 255, 255)
    border_width = 2

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for idx, bbox in enumerate(bboxes):
        x1n, y1n, x2n, y2n = bbox["bbox"]
        label = bbox["label"]

        x1 = int(x1n * img_width)
        y1 = int(y1n * img_height)
        x2 = int(x2n * img_width)
        y2 = int(y2n * img_height)

        border_color = border_colors[idx % len(border_colors)]
        draw.rectangle([x1, y1, x2, y2], outline=border_color, width=border_width)

        label_bbox = draw.textbbox((0, 0), label, font=font)
        lw = label_bbox[2] - label_bbox[0]
        lh = label_bbox[3] - label_bbox[1]
        lx = max(0, min(x1 - (lw - (x2 - x1)) // 2, img_width - lw))
        ly = max(0, y1 - lh - 10)
        draw.rectangle([lx, ly, lx + lw, ly + lh], fill=label_bg_color)
        draw.text((lx, ly), label, fill=label_text_color, font=font)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="ConsistCompose BAGEL - COCO Position Benchmark Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="sensenova/ConsistCompose-BAGEL-7B-MoT",
        help="BAGEL model path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="layout_t2i",
        choices=["layout_t2i", "layout_subject_driven", "generate", "think_generate"],
        help="BAGEL mode",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16"],
        help="Model precision type",
    )
    parser.add_argument(
        "--coco_position_benchmark_json_path",
        type=str,
        required=True,
        help="Path to COCO Position Benchmark JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval/result/coco_position/",
        help="Directory to save generated COCO benchmark images"
    )
    
    args = parser.parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"COCO JSON Path: {args.coco_position_benchmark_json_path}")
    print(f"Output Dir: {args.output_dir}")
    print("-" * 50)

    print("\nLoading ConsistCompose model...")
    model = ConsistComposeBagelModel(
        model_path=args.model_path,
        out_img_dir=args.output_dir,
        dtype=args.dtype,
    )
    set_seed()

    os.makedirs(args.output_dir, exist_ok=True)
    vis_output_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_output_dir, exist_ok=True)

    print("\nLoading COCO Position Benchmark data...")
    with open(args.coco_position_benchmark_json_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    print(f"Total COCO samples: {len(benchmark_data)}")
    print("-" * 50)

    for idx, coco_id in enumerate(benchmark_data.keys()):
        data_item = benchmark_data[coco_id]
        prompt = generate_simple_prompt(data_item)
        bbox_info = data_item["segment"]
        output_img_name = f"bagel_{coco_id}.jpg"

        print(f"[{idx+1}/{len(benchmark_data)}] COCO ID={coco_id}")
        print(f"Prompt: {prompt}")

        generated_image_path = model.generate(
            question=prompt,
            images=None,
            mode=args.mode,
            vis_bbox=False,
            resize_short_edge=768
        )

        image_output = Image.open(generated_image_path).convert("RGB")
        
        final_img_path = os.path.join(args.output_dir, output_img_name)
        image_output.save(final_img_path)

        vis_img = image_output.copy()
        if bbox_info and isinstance(bbox_info, list):
            parsed_bboxes = parse_bbox_info(bbox_info)
            if parsed_bboxes:
                vis_img = draw_bboxes_on_image(vis_img, parsed_bboxes)
        
        vis_img_path = os.path.join(vis_output_dir, output_img_name)
        vis_img.save(vis_img_path)

        print(f"✅ Saved raw image: {final_img_path}")
        print(f"✅ Saved bbox vis: {vis_img_path}")
        print("-" * 30)

    print("🎉 All COCO Position Benchmark inference done!")

if __name__ == "__main__":
    main()