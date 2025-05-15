#!/usr/bin/env python3

import torch
import json
import os
import argparse
import warnings
import numpy as np
import random
from decord import VideoReader, cpu
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image
import glob
from tqdm import tqdm
import re

warnings.filterwarnings("ignore")

# Add video processing utility functions
def extract_vision_info(conversations):
    """Extract vision information from conversations"""
    vision_infos = []
    
    if isinstance(conversations[0], list):
        # For multi-turn conversations
        for conversation in conversations:
            for message in conversation:
                if message["role"] == "user":
                    for content in message["content"]:
                        if isinstance(content, dict) and content.get("type") in ["image", "video"]:
                            vision_infos.append(content)
    else:
        # For single-turn conversations
        for message in conversations:
            if message["role"] == "user":
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") in ["image", "video"]:
                        vision_infos.append(content)
    
    return vision_infos

def fetch_video(vision_info, return_video_sample_fps=False):
    """Fetch and process video with fixed sampling of 16 or 32 frames"""
    video_path = vision_info.get("video")
    if video_path.startswith("file://"):
        video_path = video_path[7:]
    
    # Control fixed frame sampling, configurable via environment variable
    fixed_frames = int(os.environ.get('QWEN_VL_MAX_VIDEO_FRAMES', '32'))
    
    # Load video
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    video_time = total_frames / fps
    
    # print(f"Total video frames: {total_frames}, Duration: {video_time:.2f} seconds, Will sample fixed {fixed_frames} frames")
    
    # Sample fixed number of frames evenly distributed throughout the video
    if total_frames <= fixed_frames:
        # If total frames less than required sample frames, take all frames
        indices = list(range(total_frames))
    else:
        # Sample frames uniformly
        indices = np.linspace(0, total_frames-1, fixed_frames, dtype=int).tolist()
    
    # Calculate sample rate (only for return value)
    sample_fps = len(indices) / video_time if video_time > 0 else 1
    
    # Extract frames
    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in frames]
    
    if return_video_sample_fps:
        return pil_frames, sample_fps
    return pil_frames

def process_vision_info(conversations, return_video_kwargs=False):
    """Process visual information, extract images and videos"""
    vision_infos = extract_vision_info(conversations)
    
    # Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    
    for vision_info in vision_infos:
        if "image" in vision_info:
            # Image processing code
            image_path = vision_info.get("image")
            if image_path.startswith("file://"):
                image_path = image_path[7:]
            image_inputs.append(Image.open(image_path).convert('RGB'))
        elif "video" in vision_info:
            # Video processing
            video_input, video_sample_fps = fetch_video(
                vision_info, 
                return_video_sample_fps=True
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
    
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    
    if return_video_kwargs:
        return image_inputs, video_inputs, {'fps': video_sample_fps_list}
    return image_inputs, video_inputs

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    """Load video frames to maintain compatibility with old code"""
    if max_frames_num == 0:
        return [Image.new('RGB', (336, 336))], "", 0.0
    
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    
    # Special case: max_frames_num=1, use only middle frame
    if max_frames_num == 1:
        middle_frame_idx = total_frame_num // 2
        frame_idx = [middle_frame_idx]
        frame_time = [middle_frame_idx / vr.get_avg_fps()]
        frame_time_str = f"{frame_time[0]:.2f}s"
        spare_frames = vr.get_batch([middle_frame_idx]).asnumpy()
        pil_frames = [Image.fromarray(frame) for frame in spare_frames]
        
        return pil_frames, frame_time_str, video_time
    
    # Normal case: sample by fps or force sampling
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
    # Convert to PIL image list
    pil_frames = [Image.fromarray(frame) for frame in spare_frames]
    
    return pil_frames, frame_time_str, video_time

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
                f"The most accurate feedback is Option {option_num}",
                f"Final Answer: {option_num}",
                f"final answer: {option_num}",
                f"Final Answer: ({option_num})",
                f"final answer: ({option_num})"
            ]
            
            for pattern in patterns:
                if pattern in text_output:
                    selected_option = option_num
                    break
            
            if selected_option:
                break
        
        # If no pattern match, check if the last line or last few characters contain just a number
        if selected_option is None:
            # Check the last line
            lines = text_output.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                # Check if last line is just a number
                if last_line.isdigit() and 1 <= int(last_line) <= num_options:
                    selected_option = int(last_line)
            
            # If still not found, check the last few characters
            if selected_option is None:
                # Remove all whitespace and check last character
                clean_text = text_output.strip()
                if clean_text and clean_text[-1].isdigit():
                    last_digit = int(clean_text[-1])
                    if 1 <= last_digit <= num_options:
                        selected_option = last_digit
        
        # If still no match, try to extract just the number from the whole text
        if selected_option is None:
            for num in range(1, num_options + 1):
                # Use regex to find standalone numbers
                if re.search(r'\b' + str(num) + r'\b', text_output):
                    selected_option = num
                    break
        
        return selected_option
    except Exception as e:
        print(f"Error in extract_selected_option: {e}")
        return None

def load_model():
    """Load Qwen2.5-VL model and return it"""
    model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
    print(f"Loading model {model_name}...")
    
    # Set video processing parameters
    os.environ["VIDEO_MAX_PIXELS"] = str(32000 * 28 * 28 * 0.9)  # Set maximum video pixels
    
    # Load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Model loaded successfully")
    return model, processor

def analyze_video(video_path, json_item, model, processor, max_frames_num=32):
    """Analyze a single video and return results"""
    video_basename = os.path.basename(video_path)
    video_id = json_item["id"]
    
    print(f"Processing video: {video_basename} (ID: {video_id})")
    
    # Get options
    groundtruth = json_item["groundTruth"]
    options = json_item["negative_comments"]
    all_options = [groundtruth] + options
    
    # Shuffle options order
    shuffled_indices = list(range(len(all_options)))
    random.shuffle(shuffled_indices)
    shuffled_options = [all_options[i] for i in shuffled_indices]
    
    # Build prompt text
    options_text = "\n".join([f"Option {i+1}: {option}" for i, option in enumerate(shuffled_options)])
    
    # Add scenario text (if available)
    scenario_text = json_item.get("scenario_text", "")
    scenario_prompt = f"Context: {scenario_text}\n\n" if scenario_text else ""
    
    prompt_text = (
        f"{scenario_prompt}"
        f"Below are different feedback statements about the person's performance in this video:\n\n"
        f"{options_text}\n\n"
        f"Based on what you observe in the video, which option provides the most accurate feedback? "
        f"Just respond with the option number (1-{len(shuffled_options)}) and nothing else."
    )
    
    # Build message format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": f"file://{video_path}"},
                {"type": "text", "text": prompt_text}
            ],
        }
    ]
    
    # Properly process video input
    try:
        # Use official method to process video
        
        images, videos, video_kwargs = process_vision_info(
            messages, 
            return_video_kwargs=True
        )
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Process input
        inputs = processor(
            text=text,
            images=images,  # Can be None
            videos=videos,  # Video input
            return_tensors="pt",
            **video_kwargs  # Contains fps and other video parameters
        )
        
        # Move to CUDA - safely handle both tensors and non-tensor objects
        inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Inference: generate output
        with torch.no_grad():
            try:
                generated_ids = model.generate(**inputs, max_new_tokens=500)
                
                # Get input_ids from dictionary if it exists
                if 'input_ids' in inputs:
                    input_ids = inputs['input_ids']
                    # Trim generated ids by removing input tokens
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                else:
                    # If no input_ids in inputs, decode all generated tokens
                    output_text = processor.batch_decode(
                        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
            except Exception as e:
                print(f"Generation process error: {e}")
                output_text = ""
    except Exception as e:
        print(f"Video processing error: {e}")
        output_text = ""
    
    print(f"Model output: {output_text}")
    
    # Extract selected option
    selected_option = None

    for i in range(len(shuffled_options)):
        option_num = i + 1
        if f"Option {option_num}" in output_text or f"option {option_num}" in output_text or str(option_num) in output_text:
            selected_option = option_num
            break
    
    if selected_option is None:
        # Try to extract number directly from text
        for num in range(1, len(shuffled_options) + 1):
            if str(num) in output_text:
                selected_option = num
                break

    if selected_option:
        original_index = shuffled_indices[selected_option - 1]
        is_correct = (original_index == 0)  # 0 is groundTruth index
        print(f"Video: {video_basename}")
        print(f"Model selected: Option {selected_option}")
        print(f"Correct: {'Yes' if is_correct else 'No'}")
    else:
        print(f"Could not determine selected option from output: {output_text}")
        is_correct = False
    
    return {
        "video_name": video_basename,
        "video_id": video_id,
        "domain": json_item.get("domain", ""),
        "is_ge": json_item.get("is_ge", ""),
        "model_response": output_text,
        "selected_option": selected_option,
        "is_correct": is_correct,
        "groundtruth": groundtruth,
        "selected_text": shuffled_options[selected_option-1] if selected_option else None,
        "shuffled_options": shuffled_options,
        "original_indices": shuffled_indices
    }

def process_videos(video_dir, json_file, max_frames_num=32):
    """Process all videos in a directory and calculate overall accuracy"""
    # Load JSON data
    print(f"Loading options from {json_file}...")
    with open(json_file, 'r') as f:
        json_data_list = json.load(f)
    
    # Create dictionary for quick lookups by ID
    json_data_dict = {item["id"]: item for item in json_data_list}
    
    # Create backward-compatible dictionary (removing video_time)
    original_id_dict = {}
    for item in json_data_list:
        full_id = item["id"]
        # If ID contains video_time, extract original ID part
        parts = full_id.split('_')
        if len(parts) > 1 and parts[-1].replace('.', '', 1).isdigit():
            # Last part looks like a numeric value (video_time)
            original_id = '_'.join(parts[:-1])  # Everything except last part
            original_id_dict[original_id] = item
    
    # Get video files
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Load model once
    model, processor = load_model()
    
    # Process each video
    results = []
    processed_count = 0
    skipped_count = 0
    
    # Create results directory to prevent losing progress
    result_dir = os.path.join(os.path.dirname(video_dir), "qwen_results_72b_32_default")
    os.makedirs(result_dir, exist_ok=True)
    
    for video_path in tqdm(video_files):
        video_basename = os.path.basename(video_path)
        # Remove extension
        video_name_without_ext = os.path.splitext(video_basename)[0]
        
        # Check for existing results
        result_file = os.path.join(result_dir, f"{video_name_without_ext}.json")
        if os.path.exists(result_file):
            # print(f"Loading existing results for {video_basename}")
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                results.append(result)
                processed_count += 1
                continue
            except Exception as e:
                print(f"Error loading existing results: {e}, will reprocess")
        
        # Try direct lookup using filename (without extension)
        if video_name_without_ext in json_data_dict:
            # Perfect match - new style ID
            json_item = json_data_dict[video_name_without_ext]
            processed_count += 1
            result = analyze_video(
                video_path, 
                json_item, 
                model, 
                processor, 
                max_frames_num
            )
            
            # Save individual result
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            results.append(result)
            
            # Print progress every 10 items
            if processed_count % 10 == 0:
                correct_so_far = sum(1 for r in results if r.get("is_correct", False))
                print(f"Progress: {processed_count}/{len(video_files)} videos processed. "
                      f"Current accuracy: {correct_so_far/processed_count:.2%}")
                
            continue
        
        # If no direct match, try to extract original ID (for backward compatibility)
        parts = video_name_without_ext.split('_')
        if len(parts) >= 2:
            # Last part might be video_time, try without it
            potential_original_id = '_'.join(parts[:-1])
            
            if potential_original_id in original_id_dict:
                # Found match in original ID dictionary
                json_item = original_id_dict[potential_original_id]
                processed_count += 1
                result = analyze_video(
                    video_path, 
                    json_item, 
                    model, 
                    processor, 
                    max_frames_num
                )
                
                # Save individual result
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)
                    
                results.append(result)
                
                # Print progress every 10 items
                if processed_count % 10 == 0:
                    correct_so_far = sum(1 for r in results if r.get("is_correct", False))
                    print(f"Progress: {processed_count}/{len(video_files)} videos processed. "
                          f"Current accuracy: {correct_so_far/processed_count:.2%}")
                    
                continue
        
        # If we reach here, no match was found
        print(f"Skipping {video_basename}: no matching JSON data found")
        skipped_count += 1
    
    # Calculate overall accuracy
    correct_count = sum(1 for r in results if r.get("is_correct", False))
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Calculate domain-specific accuracy
    domains = set(r["domain"] for r in results if "domain" in r)
    domain_stats = {}
    for domain in domains:
        domain_results = [r for r in results if r.get("domain") == domain]
        domain_correct = sum(1 for r in domain_results if r.get("is_correct", False))
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
    
    ge_correct = sum(1 for r in ge_results if r.get("is_correct", False))
    ge_total = len(ge_results)
    ge_accuracy = ge_correct / ge_total if ge_total > 0 else 0
    
    non_ge_correct = sum(1 for r in non_ge_results if r.get("is_correct", False))
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
    output_file = os.path.join(os.path.dirname(video_dir), "qwen_video_analysis_results_72b_32_part2.json")
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

def analyze_single_video(video_path, json_file, max_frames_num=32):
    """Analyze a single video file with options from a JSON file"""
    print(f"Loading options from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create a mock json_item compatible with analyze_video
    video_id = os.path.basename(video_path).split('_')[0]  # Simplified ID extraction
    json_item = {
        "id": video_id,
        "groundTruth": data["groundTruth"],
        "negative_comments": data["negative_comments"]
    }
    
    # Load model
    model, processor = load_model()
    
    # Analyze video
    result = analyze_video(
        video_path, 
        json_item, 
        model, 
        processor, 
        max_frames_num
    )
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Analyze video options using Qwen2.5-VL model")
    parser.add_argument("--video", type=str, default="",
                       help="Path to a single video file (for individual analysis)")
    parser.add_argument("--video_dir", type=str, default="/mnt/bum/hanyi/repo/VLM/video_clips_final",
                       help="Directory containing video files (for batch processing)")
    parser.add_argument("--json", type=str, default="/mnt/bum/hanyi/repo/VLM/exact.json",
                       help="Path to JSON file containing options")
    parser.add_argument("--frames", type=int, default=32,
                       help="Fixed number of frames to sample from each video, recommended 16 or 32")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["QWEN_VL_MAX_VIDEO_FRAMES"] = str(args.frames)
    print(f"Setting fixed frame sampling: {args.frames}")
    
    if args.video and os.path.exists(args.video):
        # Analyze a single video
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found at {args.json}")
            return
        analyze_single_video(args.video, args.json, args.frames)
    elif args.video_dir and os.path.exists(args.video_dir):
        # Process multiple videos in directory
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found at {args.json}")
            return
        process_videos(args.video_dir, args.json, args.frames)
    else:
        print("Error: Either --video or --video_dir must be provided with a valid path")
        parser.print_help()

if __name__ == "__main__":
    main() 