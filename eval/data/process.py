import json
import argparse

def process_msbench_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            
            data = json.loads(line)
            num_images = len(data.get("ref_paths", []))
            image_tokens = "".join(["<image>"] * num_images)
            
            if image_tokens:
                data["prompt_grounding"] = f"{image_tokens} {data['prompt_grounding']}"
            
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input MSBench JSONL file")
    parser.add_argument("--output", required=True, help="Path to output processed JSONL file")
    args = parser.parse_args()
    
    process_msbench_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()