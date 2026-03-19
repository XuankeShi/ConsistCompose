# eval_clip_openai_only.py
# Evaluate generated images with OpenAI CLIP only.
# Metrics:
#   - CLIP-T: image-text similarity
#   - CLIP-I-local: cropped subject vs reference similarity
#   - LMS success rate: all local scores exceed threshold(s)

import os
import json
import argparse
from typing import List, Dict
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import clip


class OpenAIBackend:
    """OpenAI CLIP backend."""

    def __init__(self, model_name: str, device: torch.device):
        self.clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device.type, jit=False)
        self.model = self.model.eval()

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """Encode a list of PIL images into normalized CLIP image features."""
        feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensor_batch = torch.stack([self.preprocess(im) for im in batch])

            if self.device.type == "cuda":
                tensor_batch = tensor_batch.to(self.device, dtype=torch.float16, non_blocking=True)
            else:
                tensor_batch = tensor_batch.to(self.device)

            f = self.model.encode_image(tensor_batch)
            feats.append(f.float())

        feats = torch.cat(feats, dim=0)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        """Encode a list of texts into normalized CLIP text features."""
        feats = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tokens = self.clip.tokenize(batch, truncate=True).to(self.device)
            f = self.model.encode_text(tokens)
            feats.append(f.float())

        feats = torch.cat(feats, dim=0)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def read_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity assuming inputs are already normalized."""
    return a @ b.T


def parse_thresholds(th_str: str) -> List[float]:
    """Parse comma-separated thresholds, e.g. '0.6,0.65'."""
    return [float(x.strip()) for x in th_str.split(",") if x.strip()]

def resolve_path(path: str, root: str = None) -> str:
    """Resolve path with an optional root directory when the input path is relative."""
    if not path:
        return path
    if os.path.isabs(path) or root is None:
        return path
    return os.path.join(root, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True, help="Input JSONL aligned with generation order")
    parser.add_argument("--gen_dir", required=True, help="Directory of generated images")
    parser.add_argument("--out_csv", default="clip_scores.csv")
    parser.add_argument(
        "--ref_root",
        default=None,
        help="Root directory for reference images when paths in JSONL are relative"
    )
    parser.add_argument(
    "--model_name",
    default="ViT-B/32",
    help="OpenAI CLIP model name or checkpoint path"
    )
    parser.add_argument(
        "--pattern",
        default="{idx:05d}_{comb}.jpg",
        help="Generated filename pattern. Supported fields: idx, comb"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_batch", type=int, default=32)
    parser.add_argument("--txt_batch", type=int, default=64)
    parser.add_argument(
        "--lms_thresholds",
        default="0.6,0.65",
        help="Comma-separated LMS thresholds"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    thresholds = parse_thresholds(args.lms_thresholds)
    backend = OpenAIBackend(args.model_name, device)

    items = read_jsonl(args.jsonl)
    rows = []

    for idx, item in tqdm(list(enumerate(items)), total=len(items)):
        fn = args.pattern.format(idx=idx, comb=item.get("comb", "comb"))
        gen_path = os.path.join(args.gen_dir, fn)
        if not os.path.exists(gen_path):
            print(f"Warning: missing generated image: {gen_path}")
            continue

        gen_img = Image.open(gen_path).convert("RGB")
        width, height = gen_img.size

        # CLIP-T: generated image vs prompt
        text = item.get("prompt", "")
        if text:
            img_feat = backend.encode_images([gen_img], batch_size=args.img_batch)
            txt_feat = backend.encode_texts([text], batch_size=args.txt_batch)
            clip_t = float(cosine_sim(img_feat, txt_feat).squeeze().item())
        else:
            clip_t = 0.0

        # CLIP-I-local: cropped generated region vs aligned reference image
        bboxes = item.get("bboxes_norm", [])
        ref_paths = item.get("ref_paths", [])
        local_scores: List[float] = []

        valid_refs: List[Image.Image] = []
        ref_idx_map: List[int] = []

        for j, p in enumerate(ref_paths):
            resolved_p = resolve_path(p, args.ref_root)
            if os.path.exists(resolved_p):
                try:
                    valid_refs.append(Image.open(resolved_p).convert("RGB"))
                    ref_idx_map.append(j)
                except Exception:
                    print(f"Warning: failed to open reference image: {resolved_p}")
            else:
                print(f"Warning: missing reference image: {resolved_p}")

        ref_feats = None
        if valid_refs:
            ref_feats = backend.encode_images(valid_refs, batch_size=args.img_batch)

        for j, bbox in enumerate(bboxes):
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1p, y1p, x2p, y2p = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            x1p, y1p = max(0, x1p), max(0, y1p)
            x2p, y2p = min(width, x2p), min(height, y2p)

            if x2p <= x1p or y2p <= y1p:
                continue

            crop = gen_img.crop((x1p, y1p, x2p, y2p))
            crop_feat = backend.encode_images([crop], batch_size=1)

            if ref_feats is not None and j in ref_idx_map:
                ref_pos = ref_idx_map.index(j)
                ref_feat = ref_feats[ref_pos].unsqueeze(0)
                sim = float(cosine_sim(crop_feat, ref_feat).squeeze().item())
                local_scores.append(sim)

        clip_i_local = float(np.mean(local_scores)) if local_scores else 0.0

        # LMS success: all local scores must exceed the threshold
        success_flags = {}
        for th in thresholds:
            success_flags[f"success@{th}"] = int(bool(local_scores) and all(s >= th for s in local_scores))

        row = {
            "idx": idx,
            "comb": item.get("comb", "comb"),
            "image_file": fn,
            "clip_t": clip_t,
            "clip_i_local": clip_i_local,
            "n_local": len(local_scores),
        }
        row.update(success_flags)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved to: {args.out_csv}")

    if len(df):
        print("Summary statistics:")
        print(df[["clip_t", "clip_i_local"]].describe())

        summary = {}
        desc = df[["clip_t", "clip_i_local"]].describe()
        for col in ["clip_t", "clip_i_local"]:
            for stat in desc.index:
                summary[f"{col}_{stat}"] = desc.loc[stat, col]

        for th in thresholds:
            col = f"success@{th}"
            if col in df.columns:
                rate = float(df[col].mean())
                print(f"LMS success rate @ {th}: {rate:.4f} ({df[col].sum()}/{len(df)})")
                summary[f"LMS@{th}"] = rate
                summary[f"LMS@{th}_count"] = int(df[col].sum())
                summary[f"LMS@{th}_total"] = len(df)

        pd.DataFrame([summary]).to_csv(args.out_csv.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()