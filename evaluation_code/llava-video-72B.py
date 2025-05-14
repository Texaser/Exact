#!/usr/bin/env python3

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import json
import os
import glob
import argparse
import warnings
from decord import VideoReader, cpu
import numpy as np
import random
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:

        return np.zeros((1, 336, 336, 3)), "", 0.0
    

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    

    if max_frames_num == 1:

        middle_frame_idx = total_frame_num // 2
        frame_idx = [middle_frame_idx]
        frame_time = [middle_frame_idx / vr.get_avg_fps()]
        frame_time_str = f"{frame_time[0]:.2f}s"
        spare_frames = vr.get_batch([middle_frame_idx]).asnumpy()
        return spare_frames, frame_time_str, video_time
    

    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time_str, video_time

def load_model():
    """Load the LLaVA-Video model once and return it"""
    pretrained = "lmms-lab/LLaVA-Video-72B-Qwen2"
    cache_dir = "/mnt/bum/hanyi/hf_cache/transformers"
    model_name = "llava_qwen"
    device_map = "auto"
    
    print(f"Loading model from {pretrained}...")
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, torch_dtype="bfloat16", 
        device_map=device_map, cache_dir=cache_dir,
    )
    model.eval()
    
    print("Model loaded successfully")
    return tokenizer, model, image_processor, max_length

def analyze_video(video_path, json_item, tokenizer, model, image_processor, max_frames_num=64):
    """Analyze a single video and return the result"""
    device = "cuda"
    video_basename = os.path.basename(video_path)
    video_id = json_item["id"]
    
    print(f"Processing video: {video_basename} (ID: {video_id})")
    
    # Load video
    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
    video = [video]
    
    # Get options
    groundtruth = json_item["groundTruth"]
    options = json_item["negative_comments"]
    all_options = [groundtruth] + options
    
    # Shuffle options with fixed seed for reproducibility
    # random.seed(42) 
    shuffled_indices = list(range(len(all_options)))
    random.shuffle(shuffled_indices)
    shuffled_options = [all_options[i] for i in shuffled_indices]
    
    # Create prompt
    conv_template = "qwen_1_5"
    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}."
    
    options_text = "\n".join([f"Option {i+1}: {option}" for i, option in enumerate(shuffled_options)])
    
    # Add scenario text if available
    scenario_text = json_item.get("scenario_text", "")
    scenario_prompt = f"Context: {scenario_text}\n\n" if scenario_text else ""
    
    question = (
        # f"{DEFAULT_IMAGE_TOKEN}\n{time_instruction}\n\n"
        f"{DEFAULT_IMAGE_TOKEN}\n\n\n"
        f"{scenario_prompt}"
        f"Below are different feedback statements about the person's performance in this video:\n\n"
        f"{options_text}\n\n"
        f"Based on what you observe in the video, which option provides the most accurate feedback? "
        f"Just respond with the option number (1-{len(shuffled_options)}) and nothing else."
    )
    # print(question)
    # Create conversation and generate response
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=10,  # Reduced, since we only need a number
        )
    
    text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    # Extract the selected option number
    selected_option = None
    for i in range(len(shuffled_options)):
        option_num = i + 1
        if f"Option {option_num}" in text_output or f"option {option_num}" in text_output or str(option_num) in text_output:
            selected_option = option_num
            break
    
    if selected_option is None:
        # Try to extract just the number
        for num in range(1, len(shuffled_options) + 1):
            if str(num) in text_output:
                selected_option = num
                break
    
    if selected_option:
        original_index = shuffled_indices[selected_option - 1]
        is_correct = (original_index == 0)
        print(f"Video: {video_basename}")
        print(f"Model selected: Option {selected_option}")
        print(f"Correct: {'Yes' if is_correct else 'No'}")
    else:
        print(f"Could not determine selected option from: {text_output}")
        is_correct = False
    
    return {
        "video_name": video_basename,
        "video_id": video_id,
        "domain": json_item.get("domain", ""),
        "is_ge": json_item.get("is_ge", ""),
        "model_response": text_output,
        "selected_option": selected_option,
        "is_correct": is_correct,
        "groundtruth": groundtruth,
        "selected_text": shuffled_options[selected_option-1] if selected_option else None,
        "shuffled_options": shuffled_options,
        "original_indices": shuffled_indices
    }

def process_videos(video_dir, json_file, max_frames_num=64):
    """Process all videos in a directory and calculate overall accuracy"""
    print(f"Loading options from {json_file}...")
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Create a dictionary for faster lookups using ID as key
    json_data_dict = {item["id"]: item for item in json_data}
    
    # Also create alternate dictionary using original ID format (without video_time) for backward compatibility
    original_id_dict = {}
    for item in json_data:
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
    tokenizer, model, image_processor, _ = load_model()
    
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
            result = analyze_video(video_path, json_item, tokenizer, model, image_processor, max_frames_num)
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
                result = analyze_video(video_path, json_item, tokenizer, model, image_processor, max_frames_num)
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
    
    # Calculate GE vs non-GE accuracy
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
    
    print("\nGE vs non-GE accuracy:")
    print(f"  GE: {ge_accuracy:.2%} ({ge_correct}/{ge_total})")
    print(f"  Non-GE: {non_ge_accuracy:.2%} ({non_ge_correct}/{non_ge_total})")
    
    # Save results to file
    output_file = os.path.join(os.path.dirname(video_dir), "llava-video-72B_analysis_results_8.json")
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
    parser = argparse.ArgumentParser(description="Analyze options in videos using LLaVA-Video model")
    parser.add_argument("--video_dir", type=str, default="video_clips",
                       help="Directory containing video files")
    parser.add_argument("--json", type=str, default="exact.json",
                       help="Path to the JSON file with options")
    parser.add_argument("--frames", type=int, default=32,
                       help="Maximum number of frames to sample from each video")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found at {args.video_dir}")
        return
    
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found at {args.json}")
        return
    
    process_videos(args.video_dir, args.json, args.frames)

if __name__ == "__main__":
    main() 