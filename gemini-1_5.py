#!/usr/bin/env python3

import base64
import cv2
import os
import json
import argparse
import random
import glob
import tqdm
import numpy as np
from google import genai
from google.genai import types
import tempfile
import uuid

def process_video(video_path, max_frames=64):
    base64Frames = []
    video = cv2.VideoCapture(video_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_time = total_frame_num / fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    temp_dir = "/tmp/gemini_tmp_videos"
    os.makedirs(temp_dir, exist_ok=True)
    
    video_basename = os.path.basename(video_path)
    video_name = os.path.splitext(video_basename)[0]
    unique_id = str(uuid.uuid4())[:8]
    temp_video_path = os.path.join(temp_dir, f"{video_name}_{unique_id}_processed.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frame_time = [i/fps for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    

    for idx in frame_idx:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = video.read()
        if not success:
            break
        
        out.write(frame)
        

        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    video.release()
    out.release()
    
    print(f"Extracted {len(base64Frames)} frames from video and created processed video")
    return base64Frames, frame_time_str, video_time, temp_video_path

def analyze_video(video_path, json_item, api_key, max_frames=64):
    video_basename = os.path.basename(video_path)
    video_id = json_item["id"]
    
    print(f"Processing video: {video_basename} (ID: {video_id})")
    
    frames, frame_time, video_time, temp_video_path = process_video(video_path, max_frames)
    
    groundtruth = json_item["groundTruth"]
    options = json_item["negative_comments"]
    all_options = [groundtruth] + options
    
    shuffled_indices = list(range(len(all_options)))
    random.shuffle(shuffled_indices)
    shuffled_options = [all_options[i] for i in shuffled_indices]
    
    time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(frames)} frames are uniformly sampled from it. These frames are located at {frame_time}."
    
    options_text = "\n".join([f"Option {i+1}: {option}" for i, option in enumerate(shuffled_options)])
    
    scenario_text = json_item.get("scenario_text", "")
    scenario_prompt = f"Context: {scenario_text}\n\n" if scenario_text else ""
    
    prompt_content = (
        f"You are analyzing a video that has been sampled into individual frames. {time_instruction}\n\n"
        f"{scenario_prompt}"
        f"Below are different feedback statements about the person's performance in this video:\n\n"
        f"{options_text}\n\n"
        f"Based on what you observe in the video, which option provides the most accurate feedback? "
        f"Just respond with the option number (1-{len(shuffled_options)}) and nothing else."
    )
    
    with open(temp_video_path, 'rb') as f:
        video_bytes = f.read()
    
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        # model='models/gemini-2.5-pro-latest',
        model='models/gemini-1.5-pro',
        # model='models/gemini-2.5-pro-exp-03-25',
        contents=types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                ),
                types.Part(text=prompt_content)
            ]
        )
    )
    
    try:
        os.remove(temp_video_path)
        print(f"Removed temporary video: {temp_video_path}")
    except Exception as e:
        print(f"Failed to remove temporary video: {e}")
    
    text_output = response.text.strip()
    
    selected_option = None
    for i in range(len(shuffled_options)):
        option_num = i + 1
        if f"Option {option_num}" in text_output or f"option {option_num}" in text_output or str(option_num) in text_output:
            selected_option = option_num
            break
    
    if selected_option is None:
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

def process_videos(video_dir, json_file, api_key, max_frames=64):

    print(f"Loading options from {json_file}...")
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    

    json_data_dict = {item["id"]: item for item in json_data}
    

    original_id_dict = {}
    for item in json_data:
        full_id = item["id"]

        parts = full_id.split('_')
        if len(parts) > 1 and parts[-1].replace('.', '', 1).isdigit():

            original_id = '_'.join(parts[:-1])  
            original_id_dict[original_id] = item
    

    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    

    results = []
    processed_count = 0
    skipped_count = 0
    
    for video_path in tqdm.tqdm(video_files):
        video_basename = os.path.basename(video_path)

        video_name_without_ext = os.path.splitext(video_basename)[0]
        

        if video_name_without_ext in json_data_dict:

            json_item = json_data_dict[video_name_without_ext]
            processed_count += 1
            result = analyze_video(video_path, json_item, api_key, max_frames)
            results.append(result)
            

            if processed_count % 50 == 0:
                correct_so_far = sum(1 for r in results if r["is_correct"])
                print(f"Progress: {processed_count}/{len(video_files)} videos processed. "
                      f"Accuracy so far: {correct_so_far/processed_count:.2%}")
            
            continue
        
        
        parts = video_name_without_ext.split('_')
        if len(parts) >= 2:
            
            potential_original_id = '_'.join(parts[:-1])
            
            if potential_original_id in original_id_dict:
                
                json_item = original_id_dict[potential_original_id]
                processed_count += 1
                result = analyze_video(video_path, json_item, api_key, max_frames)
                results.append(result)
                
                
                if processed_count % 50 == 0:
                    correct_so_far = sum(1 for r in results if r["is_correct"])
                    print(f"Progress: {processed_count}/{len(video_files)} videos processed. "
                          f"Accuracy so far: {correct_so_far/processed_count:.2%}")
                
                continue
        
        
        print(f"Skipping {video_basename}: no matching JSON data found")
        skipped_count += 1
    
    
    correct_count = sum(1 for r in results if r["is_correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    
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
    
    
    output_file = os.path.join(os.path.dirname(video_dir), "gemini_analysis_results_final.json")
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
    parser = argparse.ArgumentParser(description="Analyze options in videos using Gemini model")
    parser.add_argument("--video_dir", type=str, default="video_clips",
                       help="Directory containing video files")
    parser.add_argument("--json", type=str, default="exact.json",
                       help="Path to the JSON file with options")
    parser.add_argument("--frames", type=int, default=32,
                       help="Maximum number of frames to sample from each video")
    parser.add_argument("--api_key", type=str, default=None,
                       help="Google API key")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found at {args.video_dir}")
        return
    
    if not os.path.exists(args.json):
        print(f"Error: JSON file not found at {args.json}")
        return
    
    process_videos(args.video_dir, args.json, args.api_key, args.frames)

if __name__ == "__main__":
    main()