import os
import glob
import json
import re
import torch

os.environ["HF_HOME"] = "/root/autodl-tmp/cache/huggingface"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.makedirs("/root/autodl-tmp/cache/huggingface", exist_ok=True)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

FEIX_GRASP_CANDIDATES = [
    "Large Diameter", "Small Diameter", "Medium Wrap", "Adducted Thumb", "Light Tool",
    "Power Disk", "Power Sphere", "Fixed Hook", "Index Finger Extension", "Extension Type",
    "Distal", "Palmar", "Ring", "Sphere 3 Finger", "Sphere 4 Finger",
    "Lateral", "Lateral Tripod", "Stick", "Ventral", "Adduction Grip", "Tripod Variation",
    "Palmar Pinch", "Tip Pinch", "Inferior Pincer", "Prismatic 2 Finger", "Prismatic 3 Finger",
    "Prismatic 4 Finger", "Tripod", "Quadpod", "Precision Disk", "Precision Sphere",
    "Writing Tripod", "Parallel Extension", "none"
]


def build_fig16_prompt(hand_bbox: list, obj_bbox: list, is_right: bool) -> str:
    """
    Step 1: Synthesize VLM input prompt for 6-field affordance extraction.
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
    Step 1 parser: Robust regex extractor with strict neutral fallbacks (zero hardcoding).
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


def build_grasp_taxonomy_prompt(hand_bbox: list, obj_bbox: list, is_right: bool, parsed_fields: dict) -> str:
    """
    Step 2: Feix Grasp Taxonomy classification prompt.
    """
    hand_label = "Right hand bbox" if is_right else "Left hand bbox"
    h_str = f"({int(hand_bbox[0])}, {int(hand_bbox[1])}, {int(hand_bbox[2])}, {int(hand_bbox[3])})" if hand_bbox else "none"
    o_str = f"({int(obj_bbox[0])}, {int(obj_bbox[1])}, {int(obj_bbox[2])}, {int(obj_bbox[3])})" if obj_bbox else "none"
    candidates_str = ", ".join(FEIX_GRASP_CANDIDATES)

    return f"""You are a robotic manipulation expert specializing in the Feix human grasp taxonomy.

Context:
- {hand_label}: {h_str}
- Object bbox: {o_str}
- Observed interaction: {parsed_fields.get("interaction_type", "unknown")}
- Observed object: {parsed_fields.get("object_size", "unknown")} {parsed_fields.get("object_shape", "unknown")} {parsed_fields.get("object_category", "unknown")}

Task:
Examine the image carefully and classify the hand's grasp configuration into EXACTLY ONE category from the Feix taxonomy list below.
- If the hand is NOT contacting or grasping any object (e.g., resting or approaching), output "none".
- If the hand IS grasping an object, distinguish carefully between power grasps (e.g., Large Diameter, Medium Wrap, Small Diameter, Ring) and precision grasps (e.g., Precision Disk, Tripod, Palmar Pinch) based on palm contact and thumb opposition.

Allowed categories (choose strictly one):
[{candidates_str}]

Write your answer strictly in the following two-line format without markdown, quotes, or JSON:
grasp_taxonomy: <exact category name from the list above>
reasoning: <one concise sentence explaining thumb and finger contact geometry>"""


def parse_grasp_output(text: str) -> tuple:
    """
    Step 2 parser: Strictly neutral fallbacks (zero hardcoding).
    """
    grasp_type = "unknown"
    reasoning = "unknown"

    m_grasp = re.search(r"\**grasp_taxonomy\**\s*:\s*(.*)", text, re.IGNORECASE)
    if m_grasp:
        val = m_grasp.group(1).strip().strip('* "\'.')
        if val:
            grasp_type = val

    m_reason = re.search(r"\**reasoning\**\s*:\s*(.*)", text, re.IGNORECASE)
    if m_reason:
        val = m_reason.group(1).strip().strip('* "\'')
        if val:
            reasoning = val

    return grasp_type, reasoning


def build_llm_summary_prompt(seven_fields: dict, handedness: str) -> str:
    """
    Step 3: LLM summarization prompt integrating all 7 parsed affordance elements.
    """
    prefix = f"A {handedness} hand"
    parsed_block = (
        f"1. hand_caption: {seven_fields['hand_caption']}\n"
        f"2. object_category: {seven_fields['object_category']}\n"
        f"3. object_shape: {seven_fields['object_shape']}\n"
        f"4. object_size: {seven_fields['object_size']}\n"
        f"5. interaction_type: {seven_fields['interaction_type']}\n"
        f"6. intention: {seven_fields['intention']}\n"
        f"7. grasp_taxonomy: {seven_fields['grasp_taxonomy']}"
    )

    return f"""Parsed captions:
{parsed_block}

Instructions:
Summarize the given parsed captions in a single sentence starting with "{prefix}...".
Rules:
- Integrate all known attributes smoothly (grasp taxonomy, interaction action, object size, shape, category, and intention).
- Ignore fields labeled "unknown", "undetermined", or "none" (do NOT invent a grasp or object if they are unknown/none).
- Output ONLY that single English sentence. Do not add quotation marks, bullet points, or explanations."""


def load_local_qwen(model_id: str):
    """
    Load Qwen2.5-VL from local SSD cache first to avoid network rate-limiting (HTTP 429).
    """
    try:
        print(f"[INFO] Loading {model_id} strictly from local SSD cache...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        return model, processor
    except OSError:
        print("[WARN] Some cache metadata missing locally; falling back to hf-mirror...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor


def main():
    print("=" * 70)
    print("[Module 2] End-to-End Affordance Description Pipeline")
    print("  Step 1: VLM 6-Field Extraction | Step 2: Feix Grasp | Step 3: Summary D")
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

    if isinstance(hand_data, dict):
        hand_bboxes = hand_data.get("bboxes", [])
        seq_is_right = hand_data.get("is_right", False)
        handedness_str = hand_data.get("handedness", "right" if seq_is_right else "left")
    else:
        hand_bboxes = hand_data
        seq_is_right = False
        handedness_str = "left"

    print(f"[INFO] Auto-Ingested Handedness Contract: {handedness_str.upper()} HAND (is_right={seq_is_right})")

    frame_paths = sorted(glob.glob("./data/dexycb_025_mug/frame_*.jpg"))
    num_frames = len(frame_paths)
    assert num_frames == 30, f"Expected 30 frames, found {num_frames}"

    # 2. Load Pretrained Qwen2.5-VL-7B-Instruct Once for All 3 Steps
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model, processor = load_local_qwen(model_id)
    model.eval()

    # 3. Sparse Keyframe Evaluation
    sample_indices = [0, 10, 15, 20, 29]
    print(f"[INFO] Running 3-step reasoning on keyframes: {sample_indices}")

    parsed_captions_map = {}
    keyframe_descriptions = {}

    for idx in sample_indices:
        img_path = frame_paths[idx]
        frame_hand_box = hand_bboxes[idx]
        frame_obj_box = object_bboxes[idx]
        frame_is_right = seq_is_right

        # --- Step 1: Extract 6 VLM Affordance Fields ---
        prompt_s1 = build_fig16_prompt(frame_hand_box, frame_obj_box, is_right=frame_is_right)
        msg_s1 = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt_s1}
            ]
        }]
        txt_s1 = processor.apply_chat_template(msg_s1, tokenize=False, add_generation_prompt=True)
        img_in_s1, vid_in_s1 = process_vision_info(msg_s1)
        in_s1 = processor(
            text=[txt_s1], images=img_in_s1, videos=vid_in_s1, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            gen_s1 = model.generate(**in_s1, max_new_tokens=256, do_sample=False)
            trim_s1 = [o[len(i):] for i, o in zip(in_s1.input_ids, gen_s1)]
            raw_s1 = processor.batch_decode(trim_s1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        parsed_6 = parse_vlm_output(raw_s1)

        # --- Step 2: Feix Grasp Taxonomy Classification ---
        prompt_s2 = build_grasp_taxonomy_prompt(frame_hand_box, frame_obj_box, is_right=frame_is_right, parsed_fields=parsed_6)
        msg_s2 = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": prompt_s2}
            ]
        }]
        txt_s2 = processor.apply_chat_template(msg_s2, tokenize=False, add_generation_prompt=True)
        img_in_s2, vid_in_s2 = process_vision_info(msg_s2)
        in_s2 = processor(
            text=[txt_s2], images=img_in_s2, videos=vid_in_s2, padding=True, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            gen_s2 = model.generate(**in_s2, max_new_tokens=128, do_sample=False)
            trim_s2 = [o[len(i):] for i, o in zip(in_s2.input_ids, gen_s2)]
            raw_s2 = processor.batch_decode(trim_s2, skip_special_tokens=True)[0].strip()

        grasp_type, grasp_reason = parse_grasp_output(raw_s2)

        # Merge into unified 7-field parsed caption
        parsed_7 = dict(parsed_6)
        parsed_7["grasp_taxonomy"] = grasp_type

        # --- Step 3: LLM Summarization into Single Coherent Sentence D ---
        prompt_s3 = build_llm_summary_prompt(parsed_7, handedness=handedness_str)
        msg_s3 = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_s3}]
        }]
        txt_s3 = processor.apply_chat_template(msg_s3, tokenize=False, add_generation_prompt=True)
        in_s3 = processor(text=[txt_s3], padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            gen_s3 = model.generate(**in_s3, max_new_tokens=128, do_sample=False)
            trim_s3 = [o[len(i):] for i, o in zip(in_s3.input_ids, gen_s3)]
            raw_s3 = processor.batch_decode(trim_s3, skip_special_tokens=True)[0].strip()

        description_d = raw_s3.splitlines()[0].strip().strip('"\'')

        # Store records
        parsed_captions_map[str(idx)] = {
            "frame_idx": idx,
            "raw_vlm_response": raw_s1,
            "raw_grasp_response": raw_s2,
            "grasp_reasoning": grasp_reason,
            "parsed_fields": parsed_7
        }
        keyframe_descriptions[str(idx)] = {
            "frame_idx": idx,
            "grasp_taxonomy": grasp_type,
            "affordance_description": description_d,
            "parsed_7_fields": parsed_7
        }

        print(f"\n[INFO] Frame {idx:02d} (7-Field + Summary D):")
        for k, v in parsed_7.items():
            print(f"  - {k}: {v}")
        print(f"  => Summary D: {description_d}")

    # 4. Export Both Intermediate 7-Field JSON & Final 30-Frame Descriptions JSON
    os.makedirs("results", exist_ok=True)

    out_parsed = "results/affordance_parsed_captions.json"
    with open(out_parsed, "w") as f:
        json.dump({
            "handedness": handedness_str,
            "primary_interaction_frame": 15,
            "primary_affordance_fields": parsed_captions_map["15"]["parsed_fields"],
            "all_evaluated_frames": parsed_captions_map
        }, f, indent=2)

    per_frame_map = []
    for t in range(num_frames):
        src_kf = max(k for k in sample_indices if k <= t)
        kf_entry = keyframe_descriptions[str(src_kf)]
        per_frame_map.append({
            "frame_idx": t,
            "source_keyframe": src_kf,
            "grasp_taxonomy": kf_entry["grasp_taxonomy"],
            "affordance_description": kf_entry["affordance_description"]
        })

    out_desc = "results/affordance_descriptions.json"
    with open(out_desc, "w") as f:
        json.dump({
            "handedness": handedness_str,
            "keyframes": sample_indices,
            "temporal_sharing_policy": "causal_zero_order_hold",
            "keyframe_descriptions": keyframe_descriptions,
            "per_frame_descriptions": per_frame_map
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"★ 7-Field Parsed Captions Saved to: {out_parsed}")
    print(f"★ Final Affordance Descriptions (D) Saved to: {out_desc}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()