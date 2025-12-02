import cv2 as cv
import os
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from glob import glob
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

VECTORDB_PATH = "/projects/vig/tangri/vectordb/"
DATA_ROOT = "/projects/vig/Datasets/VSI-Bench/videos"
DATASET_PATH = "nyu-visionx/VSI-Bench"
N_FRAMES = 128  # Number of evenly spaced frames to sample


def build_model():
    """Load BLIP-2 model for image captioning."""
    model_name = "Salesforce/blip2-opt-2.7b"
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = Blip2Processor.from_pretrained(model_name)
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


def sample_videos(video_path: str, n_frames: int):
    cap = cv.VideoCapture(video_path)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv.CAP_PROP_FPS)
    
    if total == 0:
        cap.release()
        print(f"⚠️ Empty or broken video: {video_path}")
        return [], [], []
    
    idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
    
    sampled_frames = []
    sampled_frame_timestamps = []
    sampled_frame_index = []
    
    for idx in idxs:
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.resize(frame, (640, 480))
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        sampled_frames.append(Image.fromarray(frame))
        sampled_frame_timestamps.append(idx / frame_rate)
        sampled_frame_index.append(int(idx))
    
    cap.release()
    return sampled_frames, sampled_frame_timestamps, sampled_frame_index


def preprocess_video(video_path: str, n_frames: int, model, processor, batch_size: int = 8):
    """Process video frames with BLIP-2 captioning in batches."""
    sampled_frames, sampled_frame_timestamps, sampled_frame_index = sample_videos(video_path, n_frames)
    sampled_frames_caption = []
    
    prompt = "Describe the main objects visible, their positions (left, right, center, background, foreground), and spatial relations. Be concise."
    
    # Process in batches for efficiency
    for i in tqdm(
        range(0, len(sampled_frames), batch_size),
        desc="Captioning frames",
        leave=False
    ):
        batch_frames = sampled_frames[i:i + batch_size]
        
        inputs = processor(
            images=batch_frames,
            return_tensors="pt",
            padding=True
        ).to(model.device, torch.float16)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )
        
        captions = processor.batch_decode(output_ids, skip_special_tokens=True)
        sampled_frames_caption.extend([c.strip() for c in captions])
    
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
    embeddings = embedding_model.encode(captions, show_progress_bar=False).tolist()
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


def setup_vectordb(vectordb_path: str, video_path: str, n_frames: int):
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
            video_file, n_frames, caption_model, processor
        )
        
        add_frames_to_db(
            collection, embedding_model, video_file,
            captions, timestamps, frame_indices
        )
    
    tqdm.write(f"\nDatabase setup complete! Total entries: {collection.count()}")
    return client, collection, embedding_model


if __name__ == "__main__":
    setup_vectordb(VECTORDB_PATH, DATA_ROOT, N_FRAMES)