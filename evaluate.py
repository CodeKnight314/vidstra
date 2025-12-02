import os
import json
import re
import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import ToTensor
import gc
import argparse
from torch.nn.parallel import DataParallel
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from preprocess import connect_vectordb

VECTORDB_PATH = "/projects/vig/tangri/vectordb/"
DATA_ROOT = "/projects/vig/Datasets/VSI-Bench/videos"
DATASET_PATH = "nyu-visionx/VSI-Bench"

def search_video_memory(collection, embedding_model, video_name, query, top_k=5):
    """Retrieve spatial memory relevant to a query for a given video."""
    
    q_emb = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"video_name": video_name}
    )

    if "documents" not in results or not results["documents"]:
        return []

    retrieved = []
    for caption, meta in zip(results["documents"][0], results["metadatas"][0]):
        retrieved.append({
            "caption": caption,
            "timestamp": meta["timestamp"],
            "frame_index": meta["frame_index"]
        })
    
    return retrieved

def sample_frames(video_path: str, n_frames: int = 8):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        print(f"⚠️ Empty or broken video: {video_path}")
        return []
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames

class VSIDataset(Dataset):
    def __init__(self, samples, completed_ids, data_root, n_frames):
        self.samples = []
        self.data_root = data_root
        self.n_frames = n_frames
        
        for row in samples:
            if f"vsi_{row['id']}" not in completed_ids:
                video_path = os.path.join(data_root, row["dataset"], f"{row['scene_name']}.mp4")
                self.samples.append((video_path, row))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, row = self.samples[idx]
        return video_path, row

def custom_collate_fn(batch):
    video_paths = [] 
    rows = [] 

    for video_path, row in batch: 
        video_paths.append(video_path)

        clean_row = {}
        for k, v in row.items() : 
            if v is None: 
                if k == 'options':
                    clean_row[k] = [] 
                else: 
                    clean_row[k] = ''
            else: 
                clean_row[k] = v
        
        rows.append(clean_row)
    
    return video_paths[0], rows[0]

def run_vsi_eval(
    model_name: str, out_dir: str, n_frames: int = 16, max_samples: int = None, 
    num_gpus: int = None, distributed: bool = False
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize distributed FIRST before any other operations
    if distributed and world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        local_rank = 0

    # Only connect to existing vectordb (don't preprocess!)
    # Load on main process first to avoid race conditions
    if is_main:
        print("📦 Connecting to vector database...")
    client, collection, embedding_model = connect_vectordb(VECTORDB_PATH)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"vsi_preds_{model_name.split('/')[-1].replace('-', '_').replace('.', '_').lower()}.json",
    )

    if is_main:
        print("📦 Loading VSI-Bench...")
    
    vsi = load_dataset(DATASET_PATH)
    split = vsi["test"]
    if max_samples:
        split = split.select(range(max_samples))

    total_questions = len(split)
    
    # Load existing results on main process
    completed_ids_list = []
    existing_results = []

    if is_main and os.path.exists(out_path):
        print(f"🔄 Found existing results at {out_path}, loading...")
        try:
            with open(out_path, "r") as f:
                existing_results = json.load(f)
                completed_ids_list = [r["id"] for r in existing_results]
            print(f"✅ Loaded {len(completed_ids_list)} completed samples.")
        except Exception as e:
            print(f"⚠️ Failed to load existing JSON ({e}), starting fresh.")

    # CRITICAL: Synchronize completed_ids across all processes
    if distributed and world_size > 1:
        # Broadcast completed_ids from main to all processes
        broadcast_list = [completed_ids_list]
        dist.broadcast_object_list(broadcast_list, src=0)
        completed_ids_list = broadcast_list[0]
    
    completed_ids = set(completed_ids_list)

    # Check if already complete (all processes check this together)
    if len(completed_ids) >= total_questions:
        if is_main:
            print(f"🎉 All {total_questions} questions already completed. Skipping.")
        if distributed and world_size > 1:
            dist.destroy_process_group()
        return

    if is_main:
        print(f"🚀 Loading model: {model_name}")
    
    from transformers import AutoProcessor, AutoModelForImageTextToText
    
    # Import custom models if available
    try:
        from src.modeling_qwen2_5_vl_with_vggt import Qwen2_5_VLForConditionalGenerationWithVGGT
        from src.modeling_qwen2_5_vl_with_memory import Qwen2_5_VLForConditionalGenerationWithMemory
        custom_models_available = True
    except ImportError:
        custom_models_available = False
        if is_main:
            print("Custom models not found, using standard AutoModel")

    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, use_fast=True
    )
    
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model with proper device_map for single or multi-GPU
    if custom_models_available:
        if model_name == "RichardGTang/Qwen2_5_VL-3B-WithVGGT":
            if distributed and world_size > 1:
                # For distributed, load on specific device
                model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                    model_name, 
                    trust_remote_code=True,
                    dtype=torch.float16
                ).to(device)
            else:
                # For single GPU or DataParallel, use device_map
                model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                    model_name, 
                    device_map="auto", 
                    trust_remote_code=True
                )
        elif model_name == "RichardGTang/Qwen2_5_VL-3B-WithMemory":
            if distributed and world_size > 1:
                model = Qwen2_5_VLForConditionalGenerationWithMemory.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    dtype=torch.float16
                ).to(device)
            else:
                model = Qwen2_5_VLForConditionalGenerationWithMemory.from_pretrained(
                    model_name, 
                    device_map="auto", 
                    trust_remote_code=True
                )
        else:
            if distributed and world_size > 1:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    dtype=torch.float16
                ).to(device)
            else:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_name, 
                    device_map="auto", 
                    trust_remote_code=True
                )
    else:
        if distributed and world_size > 1:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.float16
            ).to(device)
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name, 
                device_map="auto", 
                trust_remote_code=True
            )

    model.eval()
    
    if num_gpus and num_gpus > 1 and not distributed:
        if is_main:
            print(f"Using DataParallel with {num_gpus} GPUs")
        model = DataParallel(model, device_ids=list(range(num_gpus)))

    dataset = VSIDataset(split, completed_ids, DATA_ROOT, n_frames)
    
    if distributed and world_size > 1:
        # shuffle=False ensures consistent data split across processes
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            sampler=sampler,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            num_workers=0,
            collate_fn=custom_collate_fn
        )
    
    if is_main:
        print(f"Processing {len(dataset)} samples")
        if distributed and world_size > 1:
            print(f"Using {world_size} GPUs in distributed mode")
        elif num_gpus and num_gpus > 1:
            print(f"Using {num_gpus} GPUs with DataParallel")

    results = existing_results.copy() if is_main else []
    new_results = []
    
    pbar = tqdm(total=len(dataloader), desc=f"GPU {local_rank}", disable=not is_main)
    
    with torch.inference_mode():
        for video_path, row in dataloader:
            if not os.path.exists(video_path):
                if is_main:
                    print(f"⚠️ Missing: {video_path}")
                pbar.update(1)
                continue

            frames = sample_frames(video_path, n_frames)
            if not frames:
                pbar.update(1)
                continue

            q = row["question"]
            opts = row["options"]
            gt = row["ground_truth"]

            video_name = os.path.basename(video_path)

            query_text = f"{row['question']} {' '.join(row['options']) if row['options'] else ''}"
            memory_results = search_video_memory(
                collection, 
                embedding_model, 
                video_name, 
                query_text, 
                top_k=5
            )

            if memory_results:
                memory_text = "\n".join(
                    [f"[t={m['timestamp']:.2f}s | frame={m['frame_index']}] {m['caption']}"
                    for m in memory_results]
                )
            else:
                memory_text = "No stored memory retrieved."
            
            pre_prompt = (
                "These are frames of a video.\n"
                "Here is retrieved memory describing spatial content from earlier preprocessing:\n"
                f"{memory_text}\n\n"
            )

            if opts:
                options_str = " ".join(opts)
                prompt = f"{pre_prompt} {q} {options_str}\nAnswer with the option's letter from the given choices directly."
            else:
                prompt = f"{pre_prompt} {q}\nPlease answer the question using a single word or phrase."
            
            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": frame} for frame in frames]
                        + [{"type": "text", "text": prompt}]
                    ),
                }
            ]
            
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = processor(
                text=text_prompt,
                images=frames,
                return_tensors="pt"
            )
            
            if distributed and world_size > 1:
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            else:
                inputs = {k: v.to(model.device if hasattr(model, 'device') else 'cuda') 
                         if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            image_tensor = torch.stack([ToTensor()(frame) for frame in frames])
            image_tensor = image_tensor.unsqueeze(0)
            
            if distributed and world_size > 1:
                image_tensor = image_tensor.to(device, dtype=torch.float16)
            else:
                target_device = model.device if hasattr(model, 'device') else 'cuda'
                model_dtype = next(model.parameters()).dtype if hasattr(model, 'parameters') else torch.float16
                image_tensor = image_tensor.to(target_device, dtype=model_dtype )
            
            try:
                gen_model = model.module if hasattr(model, 'module') else model
                
                if model_name != "RichardGTang/Qwen2_5_VL-3B-WithMemory" and model_name != "RichardGTang/Qwen2_5_VL-3B-WithVGGT":
                    outputs = gen_model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=32,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                else:
                    outputs = gen_model.generate(
                        **inputs,
                        images_tensor=image_tensor,
                        do_sample=False,
                        max_new_tokens=32,
                        pad_token_id=processor.tokenizer.pad_token_id
                    )
                
                generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
                pred = processor.decode(generated_ids, skip_special_tokens=True).strip()
            except Exception as e:
                if is_main:
                    print(f"Error processing {video_path}: {e}")
                pred = ""
            
            qid = f"vsi_{row['id']}"
            
            if opts:
                pred_upper = pred.upper()
                letter_match = re.match(r'^([A-D])[\.\)\:\s]?', pred_upper)
                if letter_match:
                    pred_letter = letter_match.group(1)
                    pred_text = next(
                        (opt for opt in opts if opt.startswith(f"{pred_letter}.")),
                        pred
                    )
                else:
                    pred_text = next(
                        (opt for opt in opts if opt.split('. ', 1)[-1].lower() in pred.lower()),
                        pred
                    )
                pred_text = pred_text.strip()
                
                pred_content = pred_text.split('. ', 1)[-1].lower()
                gt_content = gt.split('. ', 1)[-1].lower() if '.' in gt else gt.lower()
                correct = pred_content == gt_content or pred_text.lower() == gt.lower()
            else:
                pred_text = pred
                try:
                    correct = abs(float(pred_text) - float(gt)) < 1e-2
                except Exception:
                    correct = pred_text.lower().strip() == gt.lower().strip()
            
            result = {
                "id": qid,
                "video": video_path,
                "question": q,
                "options": opts,
                "pred": pred_text,
                "gt": gt,
                "match": int(correct),
                "question_type": row.get("question_type", "unknown"),
                "dataset": row["dataset"],
            }
            
            new_results.append(result)
            
            if is_main and len(new_results) % 25 == 0:
                with open(out_path, "w") as f:
                    json.dump(results + new_results, f, indent=2)
            
            pbar.update(1)
            
            if len(new_results) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    
    pbar.close()
    
    # Gather results if distributed
    if distributed and world_size > 1:
        # Synchronize all processes before gathering
        dist.barrier()
        
        # Gather results from all processes
        gathered_results = [None] * world_size
        dist.all_gather_object(gathered_results, new_results)
        
        # Barrier to ensure all gathering is complete
        dist.barrier()
        
        if is_main:
            # Merge results from all GPUs, avoiding duplicates by id
            seen_ids = {r["id"] for r in existing_results}
            final_new_results = []
            for gpu_results in gathered_results:
                for result in gpu_results:
                    if result["id"] not in seen_ids:
                        seen_ids.add(result["id"])
                        final_new_results.append(result)
            new_results = final_new_results
            print(f"📊 Gathered {len(new_results)} new results from {world_size} GPUs")
    
    if is_main:
        results.extend(new_results)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Saved {len(results)} total predictions to {out_path}")
        
        correct = sum(r["match"] for r in results if "match" in r)
        total = len([r for r in results if "match" in r])
        if total > 0:
            accuracy = correct / total * 100
            print(f"📊 Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    if distributed and world_size > 1:
        dist.barrier()  # Final sync before cleanup
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=str, required=True, help="Model name or path")
    parser.add_argument("--o", type=str, required=True, help="Output directory")
    parser.add_argument("--n", default=32, type=int, help="Number of frames to sample")
    parser.add_argument("--ms", default=None, type=int, help="Max samples to process")
    parser.add_argument("--num_gpus", default=None, type=int, help="Number of GPUs for DataParallel")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    args = parser.parse_args()
    
    run_vsi_eval(args.m, args.o, args.n, args.ms, args.num_gpus, args.distributed)

"""
torchrun --nproc_per_node=4 -- evaluate.py \
    --m RichardGTang/Qwen2_5_VL-3B-WithVGGT \
    --o vsi_bench_outputs/ \
    --n 16 \
    --distributed
"""

"""
torchrun --nproc_per_node=2 -- evaluate.py \
    --m RichardGTang/Qwen2_5_VL-3B-WithMemory \
    --o vsi_bench_outputs/ \
    --num_gpus 2 \
    --n 16 \
    --ms 10 \
    --distributed
"""

"""
torchrun --nproc_per_node=2 -- evaluate.py \
    --m Qwen/Qwen2.5-VL-3B-Instruct\
    --o vsi_bench_outputs/ \
    --n 16 \
    --ms 10 \
    --distributed
"""
