import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
import json
import os
import argparse
import random
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import glob

def load_json_or_jsonl(path):
    """Load evaluation metadata from .json (list or single object) or .jsonl (LDJSON)."""
    if path.endswith('.jsonl'):
        data = []
        with open(path, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_number} in {path}: {e}")
                data.append(obj)
        return data
    else:
        with open(path, 'r') as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                return [loaded]
            return loaded

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
path = 'OpenGVLab/InternVL2_5-78B'
device_map = split_model('InternVL2_5-78B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    # load_in_8bit=True,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)


generation_config = dict(max_new_tokens=1024, do_sample=True)


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    video_time = max_frame / fps

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    frame_times = [idx / fps for idx in frame_indices]
    
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    
    return pixel_values, num_patches_list, frame_times, video_time

def extract_selected_option(text_output, num_options):
    """Attempt to extract the selected option number from the model's output text"""
    try:
        selected_option = None
        for i in range(num_options):
            option_num = i + 1
            patterns = [
                f"Option {option_num} is the most accurate",
                f"option {option_num} is the most accurate",
                f"Option {option_num} provides the most accurate",
                f"option {option_num} provides the most accurate",
                f"I believe Option {option_num}",
                f"I think Option {option_num}",
                f"My final answer is Option {option_num}",
                f"Option {option_num} would be the most accurate",
                f"I would select Option {option_num}",
                f"The most accurate feedback is Option {option_num}"
            ]
            
            for pattern in patterns:
                if pattern in text_output:
                    selected_option = option_num
                    break
            
            if selected_option:
                break
                
        # If no pattern match, try to extract just the number
        if selected_option is None:
            for num in range(1, num_options + 1):
                if str(num) in text_output:
                    selected_option = num
                    break
        
        return selected_option
    except:
        return None

def analyze_video(video_path, json_item, model, tokenizer, max_frames_num=8):
    """Analyze a single video and return the result"""
    video_basename = os.path.basename(video_path)
    video_id = json_item["id"]
    
    print(f"Processing video: {video_basename} (ID: {video_id})")
    
    # Load video
    pixel_values, num_patches_list, frame_times, video_time = load_video(
        video_path, num_segments=max_frames_num, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Get options
    groundtruth = json_item["groundTruth"]
    options = json_item["negative_comments"]
    all_options = [groundtruth] + options
    
    # Shuffle options
    shuffled_indices = list(range(len(all_options)))
    random.shuffle(shuffled_indices)
    shuffled_options = [all_options[i] for i in shuffled_indices]
    
    # Build prompt
    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(num_patches_list)} frames are uniformly sampled from it. These frames are located at {', '.join([f'{t:.2f}s' for t in frame_times])}."
    
    options_text = "\n".join([f"Option {i+1}: {option}" for i, option in enumerate(shuffled_options)])
    
    # Add scenario text if available
    scenario_text = json_item.get("scenario_text", "")
    scenario_prompt = f"Context: {scenario_text}\n\n" if scenario_text else ""
    
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    prompt = (
        video_prefix + 
        f"{time_instruction}\n\n"
        f"{scenario_prompt}"
        f"Below are different feedback statements about the person's performance in this video:\n\n"
        f"{options_text}\n\n"
        f"Based on what you observe in the video, which option provides the most accurate feedback? "
        f"Please respond with the option number (1-{len(shuffled_options)}) and nothing else."
    )
    
    # Generate response
    generation_config = dict(max_new_tokens=4096, do_sample=True)
    response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config,
                             num_patches_list=num_patches_list, history=None, return_history=True)
    
    print(f"Model response: {response}")
    
    # Extract selected option
    selected_option = extract_selected_option(response, len(shuffled_options))
    
    if selected_option is not None:
        original_index = shuffled_indices[selected_option - 1]
        is_correct = (original_index == 0)  # 0 is groundTruth index
        print(f"Video: {video_basename}")
        print(f"Model selected: Option {selected_option}")
        print(f"Correct: {'Yes' if is_correct else 'No'}")
    else:
        print(f"Could not determine selected option from response")
        is_correct = False
    
    return {
        "video_name": video_basename,
        "video_id": video_id,
        "domain": json_item.get("domain", ""),
        "is_ge": json_item.get("is_ge", ""),
        "model_response": response,
        "selected_option": selected_option,
        "is_correct": is_correct,
        "groundtruth": groundtruth,
        "selected_text": shuffled_options[selected_option-1] if selected_option else None,
        "shuffled_options": shuffled_options,
        "original_indices": shuffled_indices
    }

def load_internvl():
    """Load the InternVL model once and return it"""
    print("Loading InternVL model...")
    path = 'OpenGVLab/InternVL2_5-78B'
    device_map = split_model('InternVL2_5-78B')
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # load_in_8bit=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map=device_map).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print("Model loaded successfully")
    return model, tokenizer

def process_videos(video_dir, json_file, max_frames_num=8):
    """Process all videos in a directory and calculate overall accuracy"""
    # Load JSON data
    print(f"Loading options from {json_file}...")
    json_data_list = load_json_or_jsonl(json_file)
    
    # Create a dictionary for faster lookups using ID as key
    json_data_dict = {item["id"]: item for item in json_data_list}
    
    # Also create alternate dictionary using original ID format (without video_time) for backward compatibility
    original_id_dict = {}
    for item in json_data_list:
        full_id = item["id"]
        # If ID contains video_time, extract the original ID part
        parts = full_id.split('_')
        if len(parts) > 1 and parts[-1].replace('.', '', 1).isdigit():
            # Last part looks like a numeric value (video_time)
            original_id = '_'.join(parts[:-1])  # Everything except the last part
            original_id_dict[original_id] = item
    
    # Get video files
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Load model once
    model, tokenizer = load_internvl()
    
    # Process each video
    results = []
    processed_count = 0
    skipped_count = 0
    
    for video_path in tqdm(video_files):
        video_basename = os.path.basename(video_path)
        # Strip extension
        video_name_without_ext = os.path.splitext(video_basename)[0]
        
        # Try direct lookup with the filename without extension
        if video_name_without_ext in json_data_dict:
            # Perfect match - new style ID
            json_item = json_data_dict[video_name_without_ext]
            processed_count += 1
            result = analyze_video(video_path, json_item, model, tokenizer, max_frames_num)
            results.append(result)
            
            # Print progress every 50 items
            if processed_count % 50 == 0:
                correct_so_far = sum(1 for r in results if r["is_correct"])
                print(f"Progress: {processed_count}/{len(video_files)} videos processed. "
                      f"Accuracy so far: {correct_so_far/processed_count:.2%}")
            
            continue
        
        # If no direct match, try to extract original ID (for backwards compatibility)
        parts = video_name_without_ext.split('_')
        if len(parts) >= 2:
            # Last part might be video_time, try without it
            potential_original_id = '_'.join(parts[:-1])
            
            if potential_original_id in original_id_dict:
                # Found match in original ID dictionary
                json_item = original_id_dict[potential_original_id]
                processed_count += 1
                result = analyze_video(video_path, json_item, model, tokenizer, max_frames_num)
                results.append(result)
                
                # Print progress every 50 items
                if processed_count % 50 == 0:
                    correct_so_far = sum(1 for r in results if r["is_correct"])
                    print(f"Progress: {processed_count}/{len(video_files)} videos processed. "
                          f"Accuracy so far: {correct_so_far/processed_count:.2%}")
                
                continue
        
        # If we're here, no match was found
        print(f"Skipping {video_basename}: no matching JSON data found")
        skipped_count += 1
    
    # Calculate overall accuracy
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Calculate domain-specific accuracy
    domains = set(r["domain"] for r in results if "domain" in r)
    domain_stats = {}
    for domain in domains:
        domain_results = [r for r in results if r.get("domain") == domain]
        domain_correct = sum(1 for r in domain_results if r["is_correct"])
        domain_total = len(domain_results)
        domain_accuracy = domain_correct / domain_total if domain_total > 0 else 0
        domain_stats[domain] = {
            "total": domain_total,
            "correct": domain_correct,
            "accuracy": domain_accuracy
        }
    
    # Calculate GE vs non-GE accuracy if applicable
    ge_results = [r for r in results if r.get("is_ge") == True]
    non_ge_results = [r for r in results if r.get("is_ge") == False]
    
    ge_correct = sum(1 for r in ge_results if r["is_correct"])
    ge_total = len(ge_results)
    ge_accuracy = ge_correct / ge_total if ge_total > 0 else 0
    
    non_ge_correct = sum(1 for r in non_ge_results if r["is_correct"])
    non_ge_total = len(non_ge_results)
    non_ge_accuracy = non_ge_correct / non_ge_total if non_ge_total > 0 else 0
    
    print("\n===== RESULTS =====")
    print(f"Total videos found: {len(video_files)}")
    print(f"Videos processed: {processed_count}")
    print(f"Videos skipped: {skipped_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Overall accuracy: {accuracy:.2%}")
    
    print("\nDomain-specific accuracy:")
    for domain, stats in domain_stats.items():
        print(f"  {domain}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    if ge_total > 0 and non_ge_total > 0:
        print("\nGE vs non-GE accuracy:")
        print(f"  GE: {ge_accuracy:.2%} ({ge_correct}/{ge_total})")
        print(f"  Non-GE: {non_ge_accuracy:.2%} ({non_ge_correct}/{non_ge_total})")
    
    # Save results to file
    output_file = os.path.join(os.path.dirname(video_dir), "internvl_video_analysis_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "overall": {
                "accuracy": accuracy,
                "total_processed": processed_count,
                "total_skipped": skipped_count,
                "correct": correct_count,
            },
            "domains": domain_stats,
            "ge_stats": {
                "ge": {
                    "total": ge_total,
                    "correct": ge_correct,
                    "accuracy": ge_accuracy
                },
                "non_ge": {
                    "total": non_ge_total,
                    "correct": non_ge_correct,
                    "accuracy": non_ge_accuracy
                }
            },
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze options in videos using InternVL model")
    parser.add_argument("--video_dir", type=str, default="video_clips",
                       help="Directory containing video files")
    parser.add_argument("--json", type=str, default="exact.json",
                       help="Path to the JSON or JSONL file with options")
    parser.add_argument("--frames", type=int, default=32,
                       help="Maximum number of frames to sample from each video")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found at {args.video_dir}")
        return
    
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found at {args.json}")
        return
    
    # Process videos
    process_videos(args.video_dir, args.json, args.frames)

if __name__ == "__main__":
    main()

