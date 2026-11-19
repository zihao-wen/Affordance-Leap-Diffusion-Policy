import os
import glob
import json
import re
import cv2
import torch

os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.makedirs("/root/autodl-tmp/cache/huggingface", exist_ok=True)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def build_fig16_prompt(hand_bbox: list, obj_bbox: list, is_right: bool) -> str:
    """
    Synthesize VLM input prompt strictly aligned with Paper Figure 16 template.
    Dynamically maps handedness label based on upstream detector output.
    """
    hand_label = "Right hand bbox" if is_right else "Left hand bbox"
    hand_subject = "A right hand" if is_right else "A left hand"

    h_str = f"({int(hand_bbox[0])}, {int(hand_bbox[1])}, {int(hand_bbox[2])}, {int(hand_bbox[3])})" if hand_bbox else "none"
    o_str = f"({int(obj_bbox[0])}, {int(obj_bbox[1])}, {int(obj_bbox[2])}, {int(obj_bbox[3])})" if obj_bbox else "none"

    return f"""Instructions:
1. Use the provided detected hand and object bounding boxes (x1, y1, x2, y2) as reference to describe the interaction. If no bounding box is detected, use "none" for that field. The bounding boxes are given below:
- {hand_label}: {h_str}
- Object bbox: {o_str}

2. Focus on the following key aspects of the hand-object interaction:
- Shape and size: Describe the geometric shape of the object and its relative size to the hand.
- Contact and interaction: What physical relationship is happening between the hand and object?
- Pose and action: Describe how the hand is positioned and what action is being performed.

3. Use the following output keys only. Do not include any other fields, explanations, or comments. Each value should be brief (<= 1-2 sentences) and formatted exactly as specified.

Write your answers in the following format:

1. hand_caption: ...
2. object_category: ...
3. object_shape: ...
4. object_size: ...
5. interaction_type: ...
6. intention: ...

Do not use JSON, curly brackets, or quotation marks. Only use the format above.
Output Keys and Descriptions:

1. hand_caption
- What to include: A concise description of the primary action the hand is performing.
- Write in the form: {hand_subject} [does something] in present tense and active voice. Limit to <= 25 words.

2. object_category
- What to include: The semantic category of the object the hand is interacting with.
- Format: Use a single noun (singular). If the object cannot be identified, use "unknown".

3. object_shape
- Describe the geometric shape in simple words like "boxy", "spherical", "flat", or others. Prefer short phrases. If unclear, use "unknown".

4. object_size
- What to include: The object's size relative to the size of the hand. Use: "tiny", "small", "medium", "large", "huge", or "unknown".

5. interaction_type
- What to include: Describe the type of interaction the hand is performing with the object in a concise verb phrase.
- Focus on the action itself (e.g., grabbing, tapping, twisting).
- You may use your own wording. If uncertain, use "unknown".
Examples: "grabbing", "tapping", "pushing", "supporting", "pointing", "twisting", "pressing a button", "using as a tool", "resting on", "unknown"

6. intention
- What to include: The inferred high-level goal of the hand. Start with "to". Use <= 10 words. If unclear, write "undetermined"."""

def parse_vlm_output(text: str) -> dict:
    """
    Robust regex extractor with strict neutral fallbacks (zero hardcoding).
    """
    fields = {
        "hand_caption": "unknown",
        "object_category": "unknown",
        "object_shape": "unknown",
        "object_size": "unknown",
        "interaction_type": "unknown",
        "intention": "undetermined"
    }

    patterns = {
        "hand_caption": r"(?:1\.\s*)?\**hand_caption\**\s*:\s*(.*)",
        "object_category": r"(?:2\.\s*)?\**object_category\**\s*:\s*(.*)",
        "object_shape": r"(?:3\.\s*)?\**object_shape\**\s*:\s*(.*)",
        "object_size": r"(?:4\.\s*)?\**object_size\**\s*:\s*(.*)",
        "interaction_type": r"(?:5\.\s*)?\**interaction_type\**\s*:\s*(.*)",
        "intention": r"(?:6\.\s*)?\**intention\**\s*:\s*(.*)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip().strip('* "\'.')
            if val:
                fields[key] = val
    return fields

def main():
    print("=" * 70)
    print("[Module 2 - Step 1] Affordance Common-Sense Extraction via Qwen2.5-VL")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Hardware platform: {device}")

    # 1. Load Symmetrical Perception Data & Ingest Handedness Automatically
    hand_path = "results/detected_hand_bboxes.json"
    obj_path = "results/detected_object_bboxes.json"
    assert os.path.exists(hand_path), f"Missing hand bboxes at {hand_path}"
    assert os.path.exists(obj_path), f"Missing object bboxes at {obj_path}"

    with open(hand_path, "r") as f:
        hand_data = json.load(f)
    with open(obj_path, "r") as f:
        object_bboxes = json.load(f)

    # Ingest Hand BBoxes and Handedness from Data Contract
    if isinstance(hand_data, dict):
        hand_bboxes = hand_data.get("bboxes", [])
        seq_is_right = hand_data.get("is_right", False)
        handedness_str = hand_data.get("handedness", "right" if seq_is_right else "left")
    else:
        # Backward compatibility fallback
        hand_bboxes = hand_data
        seq_is_right = False
        handedness_str = "left"

    print(f"[INFO] Auto-Ingested Handedness Contract: {handedness_str.upper()} HAND (is_right={seq_is_right})")

    frame_paths = sorted(glob.glob("./data/dexycb_025_mug/frame_*.jpg"))
    assert len(frame_paths) == 30, f"Expected 30 frames, found {len(frame_paths)}"

    # 2. Load Pretrained Qwen2.5-VL-7B-Instruct
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"[INFO] Initializing {model_id} in bfloat16 precision...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 3. Subsample Representative Interaction Frames (Paper Sec 4)
    sample_indices = [0, 10, 15, 20, 29]
    print(f"[INFO] Running physical reasoning on keyframes: {sample_indices}")

    parsed_captions_map = {}

    for idx in sample_indices:
        img_path = frame_paths[idx]
        frame_hand_box = hand_bboxes[idx]
        frame_obj_box = object_bboxes[idx]
        frame_is_right = seq_is_right

        prompt_text = build_fig16_prompt(frame_hand_box, frame_obj_box, is_right=frame_is_right)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        parsed_fields = parse_vlm_output(response_text)
        parsed_captions_map[str(idx)] = {
            "frame_idx": idx,
            "raw_response": response_text,
            "parsed_fields": parsed_fields
        }

        print(f"\n[INFO] Frame {idx:02d} Inferred Common-Sense Affordances:")
        for k, v in parsed_fields.items():
            print(f"  - {k}: {v}")

    # Primary interaction state defined at keyframe 15
    os.makedirs("results", exist_ok=True)
    out_file = "results/affordance_parsed_captions.json"
    with open(out_file, "w") as f:
        json.dump({
            "handedness": handedness_str,
            "primary_interaction_frame": 15,
            "primary_affordance_fields": parsed_captions_map["15"]["parsed_fields"],
            "all_evaluated_frames": parsed_captions_map
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"★ Affordance Parsed Captions Successfully Saved to: {out_file}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()