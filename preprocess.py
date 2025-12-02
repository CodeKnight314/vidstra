import cv2 as cv
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from glob import glob
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VECTORDB_PATH = "/projects/vig/tangri/vectordb.chroma"
DATA_ROOT = "/projects/vig/Datasets/VSI-Bench/videos"
DATASET_PATH = "nyu-visionx/VSI-Bench"
FPS = 2

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
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    else:
        image = image_file.convert('RGB') if image_file.mode != 'RGB' else image_file
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def build_model():
    path = 'OpenGVLab/InternVL2-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer

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

def preprocess_video(video_path: str, fps: int, model: AutoModel, tokenizer: AutoTokenizer):
    sampled_frames, sampled_frame_timestamps, sampled_frame_index = sample_videos(video_path, fps)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    sampled_frames_caption = []

    for frame, timestamp, index in zip(sampled_frames, sampled_frame_timestamps, sampled_frame_index):
        frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        pixel_values = load_image(frame_pil, input_size=448, max_num=12).to(torch.bfloat16).cuda()
        
        question = """<image>\n
        You are generating a structured spatial description of this frame.
        Describe:
        1. The main objects visible.
        2. Their approximate positions in the frame (left, right, center, background, foreground).
        3. Any meaningful spatial relations (X is left of Y, X is behind Y, X is closer than Y).
        Keep the description short but spatially detailed.
        """

        response = model.chat(tokenizer, pixel_values, question, generation_config)
        sampled_frames_caption.append(response)

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
    vlm_model, tokenizer = build_model()
    
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
            video_file, fps, vlm_model, tokenizer
        )
        
        add_frames_to_db(
            collection, embedding_model, video_file,
            captions, timestamps, frame_indices
        )
    
    tqdm.write(f"\nDatabase setup complete! Total entries: {collection.count()}")
    return client, collection, embedding_model

if __name__ == "__main__":
    setup_vectordb(VECTORDB_PATH, DATA_ROOT, FPS)