#!/usr/bin/env python3

import torch
import json
import os
import argparse
import warnings
import numpy as np
import random
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor
import glob
import sys
from tqdm import tqdm

warnings.filterwarnings("ignore")

_original_default = json.JSONEncoder.default
def _patched_default(self, obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    return _original_default(self, obj)

json.JSONEncoder.default = _patched_default

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

def get_video_duration(video_path):

    try:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        return video_time
    except Exception as e:

        return 10.0  

def load_model():
    """Load the VideoLLaMA3 model once and return it"""
    model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"
    print(f"Loading model from {model_name}...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        print("Model loaded successfully")
        return model, processor
    except Exception as e:
        print(f"Load model failed: {e}")
        raise

def analyze_video(video_path, json_item, model, processor, max_frames_num=64):
    """Analyze a single video and return the result"""
    video_basename = os.path.basename(video_path)
    video_id = json_item["id"]
    
    print(f"Processing video: {video_basename} (ID: {video_id})")
    
    try:
        
        video_time = get_video_duration(video_path)
        
        # Get options
        groundtruth = json_item["groundTruth"]
        options = json_item["negative_comments"]
        all_options = [groundtruth] + options
        
        # Shuffle options
        shuffled_indices = list(range(len(all_options)))
        random.shuffle(shuffled_indices)
        shuffled_options = [all_options[i] for i in shuffled_indices]
        
        # Build prompt
        time_instruction = f"The video lasts for {video_time:.2f} seconds."
        
        options_text = "\n".join([f"Option {i+1}: {option}" for i, option in enumerate(shuffled_options)])
        
        # Add scenario text if available
        scenario_text = json_item.get("scenario_text", "")
        scenario_prompt = f"Context: {scenario_text}\n\n" if scenario_text else ""
        
        question = (
            f"{scenario_prompt}"
            f"Below are different feedback statements about the person's performance in this video:\n\n"
            f"{options_text}\n\n"
            f"Based on what you observe in the video, which option provides the most accurate feedback? "
            f"Just respond with the option number (1-{len(shuffled_options)}) and nothing else."
        )
        
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "max_frames": max_frames_num}},
                    # {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": max_frames_num}},
                    {"type": "text", "text": question},
                ]
            },
        ]
        
        try:
            
            inputs = processor(conversation=conversation, return_tensors="pt")
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            
            
            output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0, do_sample=False)
            output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            print(f"Model response: {output_text}")
            
        except Exception as e:
            print(f"Error in analyze_video for {video_id}: {str(e)}")
            
            try:
                print("Trying alternative method...")
                
                abs_video_path = os.path.abspath(video_path)
                
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video_path": abs_video_path},
                            {"type": "text", "text": question},
                        ]
                    },
                ]
                
                inputs = processor(conversation=conversation, return_tensors="pt")
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
                output_ids = model.generate(**inputs, max_new_tokens=512, temperature=0, do_sample=False)
                output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                print(f"Alternative method response: {output_text}")
                
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
                return {
                    "video_name": video_basename,
                    "video_id": video_id,
                    "domain": json_item.get("domain", ""),
                    "is_ge": json_item.get("is_ge", ""),
                    "error": f"{str(e)}; Alternative method: {str(e2)}",
                    "is_correct": False
                }
        
        # Extract selected option
        selected_option = extract_selected_option(output_text, len(shuffled_options))
        
        if selected_option:
            original_index = shuffled_indices[selected_option - 1]
            is_correct = (original_index == 0)  # 0 is groundTruth index
            print(f"Video: {video_basename}")
            print(f"Model selected: Option {selected_option}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")
        else:
            print(f"Could not determine selected option from: {output_text}")
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
    except Exception as e:
        print(f"Error in analyze_video for {video_id}: {str(e)}")
        return {
            "video_name": video_basename,
            "video_id": video_id,
            "domain": json_item.get("domain", ""),
            "is_ge": json_item.get("is_ge", ""),
            "error": str(e),
            "is_correct": False
        }

def process_videos(video_dir, json_file, max_frames_num=64, max_items=0):
    """Process all videos in a directory and calculate overall accuracy"""
    # Load JSON data
    print(f"Loading options from {json_file}...")
    with open(json_file, 'r') as f:
        json_data_list = json.load(f)
    
    # Apply max_items limit if specified
    if max_items > 0 and max_items < len(json_data_list):
        print(f"Limiting to {max_items} items (out of {len(json_data_list)})")
        json_data_list = json_data_list[:max_items]
    
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
    model, processor = load_model()
    
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
            result = analyze_video(video_path, json_item, model, processor, max_frames_num)
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
                result = analyze_video(video_path, json_item, model, processor, max_frames_num)
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
    domains = set(r["domain"] for r in results if "domain" in r and r["domain"])
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
    output_file = os.path.join(os.path.dirname(video_dir), "videollama_analysis_results.json")
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

def analyze_single_video(video_path, json_file, max_frames_num=64):
    """Analyze a single video file with options from a JSON file"""
    print(f"Loading options from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    video_id = os.path.basename(video_path).split('_')[0]  
    
    matching_item = None
    for item in data:
        if item.get("id") == video_id:
            matching_item = item
            break
    
    if not matching_item:
        print(f"Error: No matching data found for video ID {video_id}")
        return None
    
    
    model, processor = load_model()
    
    
    result = analyze_video(video_path, matching_item, model, processor, max_frames_num)
    
    
    output_file = os.path.splitext(video_path)[0] + "_result.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {output_file}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Analyze options in videos using VideoLLaMA3 model")
    parser.add_argument("--video", type=str, default="",
                       help="Path to a single video file (for individual analysis)")
    parser.add_argument("--video_dir", type=str, default="video_clips",
                       help="Directory containing video files (for batch processing)")
    parser.add_argument("--json", type=str, default="exact.json",
                       help="Path to the JSON file with options")
    parser.add_argument("--frames", type=int, default=32,
                       help="Maximum number of frames to sample from each video")
    parser.add_argument("--max_items", type=int, default=0,
                       help="Maximum number of items to process (0 = all)")
    
    args = parser.parse_args()
    
    print(f"Video frames: {args.frames}")
    print(f"Processing JSON file: {args.json}")
    print(f"Maximum number of items to process: {args.max_items if args.max_items > 0 else 'all'}")
    
    if args.video and os.path.exists(args.video):
        # Analyze a single video
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found at {args.json}")
            return
        print(f"Processing single video: {args.video}")
        analyze_single_video(args.video, args.json, args.frames)
    elif args.video_dir and os.path.exists(args.video_dir):
        # Process multiple videos in a directory
        if not os.path.exists(args.json):
            print(f"Error: JSON file not found at {args.json}")
            return
        print(f"Processing video directory: {args.video_dir}")
        process_videos(args.video_dir, args.json, args.frames, args.max_items)
    else:
        print("Error: Either --video or --video_dir must be provided with a valid path")
        parser.print_help()

if __name__ == "__main__":
    main() 