import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont

import torch
import random
import numpy as np

from consist_compose import ConsistComposeBagelModel

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl(file_path):
    data = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                data.append(sample)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in line {line_num}: {e}") from e
    if not data:
        raise ValueError(f"Empty JSONL file: {file_path}")
    print(f"Successfully loaded {len(data)} samples from {file_path}")
    return data

def read_jsonl(file_path):
    return load_jsonl(file_path)

def build_image_list_from_ref_paths(ref_paths, target_size):
    image_list = []
    for path in ref_paths:
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize((target_size, target_size))
            image_list.append(img)
    return image_list

def compose_bbox_info_from_tokens_and_boxes(tokens, bboxes_norm):
    return list(zip(tokens, bboxes_norm)) if tokens and bboxes_norm else []

def parse_bbob_info(vis_info):
    return [box for _, box in vis_info if len(box) == 4]

def draw_bboxes_on_image(image: Image.Image, bboxes: list, labels: list = None) -> Image.Image:
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
        x1n, y1n, x2n, y2n = bbox
        label = labels[idx] if labels and idx < len(labels) else f"object{idx+1}"

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

class timer_context:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        self.end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start:
            self.start.record()
    def __exit__(self, *args):
        if self.end:
            self.end.record()
            torch.cuda.synchronize()
            print(f"[Timer] {self.name}: {self.start.elapsed_time(self.end):.2f}ms")


def run_msbench_inference(
    model: ConsistComposeBagelModel,  # 替换为新模型
    args: argparse.Namespace
) -> None:
    """Run MS-Bench inference with ConsistComposeBagelModel"""
    vis_output_dir = os.path.join(args.out_img_dir, "vis")
    os.makedirs(vis_output_dir, exist_ok=True)
    lines = read_jsonl(args.msbench_jsonl_path)

    success_count = 0
    fail_count = 0

    for idx, item in enumerate(lines):
        prompt = (item.get("prompt_grounding") or "").strip()
        ref_paths = item.get("ref_paths", [])
        tokens = item.get("tokens_mapped", [])
        bboxes_norm = item.get("bboxes_norm", [])
        comb_type = item.get("comb", "comb")

        ref_paths = [
            p if os.path.isabs(p) else os.path.normpath(os.path.join(args.msbench_input_dir, p))
            for p in ref_paths
        ]

        print(f"\n[MS-Bench] Sample {idx}")
        print(f"Prompt: {prompt}")
        print(f"Reference Images: {ref_paths}")

        image_list = build_image_list_from_ref_paths(ref_paths, 512)
        if not image_list:
            print(f"⚠️ Sample {idx} skipped: No valid reference images")
            fail_count += 1
            continue

        try:
            with timer_context(f"msbench_inference_{idx}"):
                generated_image_path = model.generate(
                    question=prompt,
                    images=ref_paths,
                    mode=args.mode,
                    vis_bbox=False,
                )

            if not generated_image_path or not os.path.exists(generated_image_path):
                raise Exception("Model returned empty image path")

            image_output = Image.open(generated_image_path).convert("RGB")
            out_name = f"{idx:05d}_{comb_type}.jpg"
            out_path = os.path.join(args.out_img_dir, out_name)
            image_output.save(out_path)
            # ==================================================================

            # ===================== image combination show =====================

            vis_info = compose_bbox_info_from_tokens_and_boxes(tokens, bboxes_norm)
            parsed_bboxes = parse_bbob_info(vis_info)
            vis_img = image_output.copy()
            if parsed_bboxes and tokens:
                vis_img = draw_bboxes_on_image(vis_img, parsed_bboxes, tokens)

            target_height = 448
            tagged_ref_images = []
            for i, img in enumerate(image_list, 1):
                aspect_ratio = img.width / img.height
                target_width = int(aspect_ratio * target_height)
                resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                
                draw = ImageDraw.Draw(resized_img)
                tag_text = f"img{i}"
                tag_padding = 5
                try:
                    tag_font = ImageFont.truetype("arial.ttf", 16)
                except IOError:
                    tag_font = ImageFont.load_default()

                text_bbox = draw.textbbox((0,0), tag_text, font=tag_font)
                text_width = text_bbox[2] - text_bbox[0]
                tag_x = resized_img.width - text_width - tag_padding
                tag_y = tag_padding

                tag_bg = Image.new('RGBA', resized_img.size, (0,0,0,0))
                bg_draw = ImageDraw.Draw(tag_bg)
                bg_draw.rectangle([tag_x-tag_padding, tag_y-tag_padding, 
                                   tag_x+text_width+tag_padding, tag_y+(text_bbox[3]-text_bbox[1])+tag_padding], 
                                  fill=(255,255,255,180))

                resized_img = resized_img.convert('RGBA')
                tagged_img = Image.alpha_composite(resized_img, tag_bg)
                draw = ImageDraw.Draw(tagged_img)
                draw.text((tag_x, tag_y), tag_text, font=tag_font, fill=(0,0,0))
                tagged_ref_images.append(tagged_img.convert('RGB'))

            aspect_ratio = vis_img.width / vis_img.height
            target_width = int(aspect_ratio * target_height)
            resized_vis = vis_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(resized_vis)
            tag_text = "target img"
            tag_padding = 5
            try:
                tag_font = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                tag_font = ImageFont.load_default()

            text_bbox = draw.textbbox((0,0), tag_text, font=tag_font)
            text_width = text_bbox[2] - text_bbox[0]
            tag_x = resized_vis.width - text_width - tag_padding
            tag_y = tag_padding

            tag_bg = Image.new('RGBA', resized_vis.size, (0,0,0,0))
            bg_draw = ImageDraw.Draw(tag_bg)
            bg_draw.rectangle([tag_x-tag_padding, tag_y-tag_padding,
                               tag_x+text_width+tag_padding, tag_y+(text_bbox[3]-text_bbox[1])+tag_padding],
                              fill=(255,215,0,180))

            resized_vis = resized_vis.convert('RGBA')
            tagged_target = Image.alpha_composite(resized_vis, tag_bg)
            draw = ImageDraw.Draw(tagged_target)
            draw.text((tag_x, tag_y), tag_text, font=tag_font, fill=(0,0,0))
            tagged_target = tagged_target.convert('RGB')

            total_width = sum(img.width for img in tagged_ref_images) + tagged_target.width
            combined_img = Image.new('RGB', (total_width, target_height), (255,255,255))
            current_x = 0
            for img in tagged_ref_images:
                combined_img.paste(img, (current_x, 0))
                current_x += img.width
            combined_img.paste(tagged_target, (current_x, 0))

            combined_out_path = os.path.join(vis_output_dir, f"{idx:05d}_{comb_type}_combined.jpg")
            combined_img.save(combined_out_path)
            print(f"✅ Sample {idx} success!")
            success_count += 1

        except Exception as e:
            fail_count += 1
            print(f"❌ Sample {idx} failed: {str(e)}")
            continue

    print("\n" + "="*60)
    print("MS-Bench Inference Summary")
    print(f"Total samples: {len(lines)}")
    print(f"Success: {success_count} | Failed: {fail_count}")
    print(f"Output dir: {os.path.abspath(args.out_img_dir)}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="ConsistCompose Batch Inference + MS-Bench Test"
    )
    parser.add_argument("--model_path", type=str, default="sensenova/ConsistCompose-BAGEL-7B-MoT")
    parser.add_argument("--mode", type=str, default="layout_subject_driven", 
                        choices=["layout_t2i", "layout_subject_driven", "generate", "think_generate"])
    parser.add_argument("--out_img_dir", type=str, default="./output_images/msbench/")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16"])
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--msbench_jsonl_path", type=str, required=True, 
                        help="Path to MS-Bench JSONL file")
    parser.add_argument("--msbench_input_dir", type=str, required=True,
                        help="MS-Bench reference images root directory")
    args = parser.parse_args()

    os.makedirs(args.out_img_dir, exist_ok=True)
    set_seed(args.seed)

    print("\nLoading ConsistCompose model...")
    model = ConsistComposeBagelModel(
        model_path=args.model_path,
        out_img_dir=args.out_img_dir,
        dtype=args.dtype,
    )

    print("\nStarting MS-Bench Inference...")
    run_msbench_inference(model, args)

if __name__ == "__main__":
    main()