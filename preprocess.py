import cv2 as cv
import os
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from glob import glob
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

VECTORDB_PATH = "/projects/vig/tangri/vectordb.chroma"
DATA_ROOT = "/projects/vig/Datasets/VSI-Bench/videos"
DATASET_PATH = "nyu-visionx/VSI-Bench"
FPS = 2

def build_model():
    """Load Qwen3-VL model for image captioning."""
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def build_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load sentence transformer for text embeddings."""
    return SentenceTransformer(model_name)

def setup_chromadb(db_path: str, collection_name: str = "video_frames"):
    """Initialize persistent ChromaDB."""
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

def sample_videos(video_path: str, fps: int):
    video = cv.VideoCapture(video_path)
    
    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv.CAP_PROP_FPS))
    frame_interval = max(1, frame_rate // fps)
    
    sampled_frames, sampled_frame_timestamps, sampled_frame_index = [], [], []
    
    for i in range(0, frame_count, frame_interval):
        video.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if not ret:
            break
        sampled_frames.append(frame)
        sampled_frame_timestamps.append(i / frame_rate)
        sampled_frame_index.append(i)
    
    video.release()
    return sampled_frames, sampled_frame_timestamps, sampled_frame_index

def preprocess_video(video_path: str, fps: int, model, processor):
    sampled_frames, sampled_frame_timestamps, sampled_frame_index = sample_videos(video_path, fps)
    sampled_frames_caption = []

    for frame, timestamp, index in tqdm(
        zip(sampled_frames, sampled_frame_timestamps, sampled_frame_index),
        total=len(sampled_frames),
        desc="Captioning frames",
        leave=False
    ):
        frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        
        # Qwen3-VL chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_pil},
                    {"type": "text", "text": "Describe the main objects visible, their approximate positions in the frame (left, right, center, background, foreground), and any meaningful spatial relations. Keep the description short but spatially detailed."}
                ]
            }
        ]
        
        # Apply chat template and process
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[frame_pil],
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )
        
        # Decode only the generated tokens (exclude input)
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        sampled_frames_caption.append(caption)

    return sampled_frames_caption, sampled_frame_timestamps, sampled_frame_index

def add_frames_to_db(
    collection,
    embedding_model: SentenceTransformer,
    video_path: str,
    captions: list[str],
    timestamps: list[float],
    frame_indices: list[int]
):
    video_name = os.path.basename(video_path)
    embeddings = embedding_model.encode(captions, show_progress_bar=True).tolist()
    ids = [f"{video_name}_frame_{idx}" for idx in frame_indices]
    metadatas = [
        {
            "video_path": video_path,
            "video_name": video_name,
            "frame_index": int(idx),
            "timestamp": float(ts),
            "caption": caption
        }
        for idx, ts, caption in zip(frame_indices, timestamps, captions)
    ]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=captions
    )

def connect_vectordb(vectordb_path: str):
    """Connect to existing ChromaDB and load embedding model for queries only."""
    embedding_model = build_embedding_model()
    client, collection = setup_chromadb(vectordb_path)
    return client, collection, embedding_model

def setup_vectordb(vectordb_path: str, video_path: str, fps: int):
    """Full preprocessing: loads VLM, processes videos, and stores in ChromaDB."""
    caption_model, processor = build_model()
    embedding_model = build_embedding_model()
    client, collection = setup_chromadb(vectordb_path)
    
    video_files = glob(os.path.join(video_path, "**/*.mp4"), recursive=True)
    tqdm.write(f"Found {len(video_files)} video files")
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        tqdm.write(f"\nProcessing: {video_file}")
        
        video_name = os.path.basename(video_file)
        existing = collection.get(where={"video_name": video_name})
        if existing and len(existing["ids"]) > 0:
            tqdm.write(f"Skipping {video_name} - already in database")
            continue
        
        captions, timestamps, frame_indices = preprocess_video(
            video_file, fps, caption_model, processor
        )
        
        add_frames_to_db(
            collection, embedding_model, video_file,
            captions, timestamps, frame_indices
        )
    
    tqdm.write(f"\nDatabase setup complete! Total entries: {collection.count()}")
    return client, collection, embedding_model

if __name__ == "__main__":
    setup_vectordb(VECTORDB_PATH, DATA_ROOT, FPS)