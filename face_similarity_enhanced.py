import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import warnings
import logging
from pathlib import Path
from typing import Union, Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import io
import base64
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import mediapipe as mp

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    FACENET = "facenet"
    ARCFACE = "arcface"
    VGGFACE = "vggface"
    ENSEMBLE = "ensemble"


class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


@dataclass
class FaceData:
    aligned_face: np.ndarray
    embedding: np.ndarray
    confidence: float
    bbox: List[float]
    landmarks: Optional[Dict[str, np.ndarray]] = None
    quality_score: float = 0.0
    face_angle: Dict[str, float] = None


@dataclass
class SimilarityResult:
    similarity_score: float
    is_same_person: bool
    confidence: float
    distance_metric: str
    model_used: str
    threshold_used: float
    face1_quality: float
    face2_quality: float
    details: Dict[str, Any] = None


class FaceQualityAssessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
    
    def assess_quality(self, face_image: np.ndarray) -> Dict[str, float]:
        h, w = face_image.shape[:2]
        
        quality_metrics = {
            'resolution': min(h, w) / 160.0,
            'brightness': np.mean(face_image) / 255.0,
            'contrast': np.std(face_image) / 128.0,
            'sharpness': self._calculate_sharpness(face_image),
            'symmetry': self._calculate_symmetry(face_image),
            'pose_quality': self._assess_pose(face_image)
        }
        
        overall_quality = np.mean(list(quality_metrics.values()))
        quality_metrics['overall'] = overall_quality
        
        return quality_metrics
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return min(variance / 500.0, 1.0)
    
    def _calculate_symmetry(self, image: np.ndarray) -> float:
        mid = image.shape[1] // 2
        left = image[:, :mid]
        right = cv2.flip(image[:, mid:], 1)
        
        min_width = min(left.shape[1], right.shape[1])
        left = left[:, :min_width]
        right = right[:, :min_width]
        
        diff = np.mean(np.abs(left.astype(float) - right.astype(float)))
        symmetry = 1.0 - (diff / 255.0)
        return symmetry
    
    def _assess_pose(self, image: np.ndarray) -> float:
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return 0.5
        
        landmarks = results.multi_face_landmarks[0]
        nose_tip = landmarks.landmark[1]
        nose_bridge = landmarks.landmark[6]
        
        pitch = abs(nose_tip.y - nose_bridge.y)
        yaw = abs(nose_tip.x - 0.5)
        
        pose_score = 1.0 - (pitch + yaw)
        return max(0, min(1, pose_score))


class FacePreprocessor:
    def __init__(self):
        self.target_size = (160, 160)
        self.quality_assessor = FaceQualityAssessor()
        
    def preprocess(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        if enhance:
            image = self._enhance_image(image)
        
        image = self._normalize_lighting(image)
        image = self._align_and_crop(image)
        
        return image
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean = np.mean(gray)
        
        if mean < 80:
            alpha = 1.5
            beta = 30
        elif mean > 180:
            alpha = 0.8
            beta = -20
        else:
            return image
        
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted
    
    def _align_and_crop(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if h != self.target_size[0] or w != self.target_size[1]:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_CUBIC)
        return image


class EmbeddingCache:
    def __init__(self, cache_dir: Path = Path("./cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        
    def _get_hash(self, image_path: str) -> str:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def get(self, image_path: str, model_type: str) -> Optional[np.ndarray]:
        cache_key = f"{self._get_hash(image_path)}_{model_type}"
        
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)
                self.memory_cache[cache_key] = embedding
                return embedding
        
        return None
    
    def set(self, image_path: str, model_type: str, embedding: np.ndarray):
        cache_key = f"{self._get_hash(image_path)}_{model_type}"
        self.memory_cache[cache_key] = embedding
        
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)


class EnhancedFaceSimilarity:
    def __init__(self, 
                 model_type: ModelType = ModelType.ENSEMBLE,
                 use_gpu: bool = True,
                 cache_embeddings: bool = True):
        
        self.model_type = model_type
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.preprocessor = FacePreprocessor()
        self.quality_assessor = FaceQualityAssessor()
        
        if cache_embeddings:
            self.cache = EmbeddingCache()
        else:
            self.cache = None
        
        self.models = self._load_models()
        
        self.thresholds = {
            ModelType.FACENET: 0.70,
            ModelType.ARCFACE: 0.65,
            ModelType.VGGFACE: 0.75,
            ModelType.ENSEMBLE: 0.72
        }
        
        logger.info(f"Initialized with {model_type.value} model on {self.device}")
    
    def _load_models(self) -> Dict[str, Any]:
        models = {}
        
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            
            models['mtcnn'] = MTCNN(
                image_size=160,
                margin=40,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device,
                keep_all=False
            )
            
            if self.model_type in [ModelType.FACENET, ModelType.ENSEMBLE]:
                models['facenet'] = InceptionResnetV1(
                    pretrained='vggface2'
                ).eval().to(self.device)
            
            if self.model_type in [ModelType.VGGFACE, ModelType.ENSEMBLE]:
                models['vggface'] = InceptionResnetV1(
                    pretrained='casia-webface'
                ).eval().to(self.device)
                
        except ImportError as e:
            logger.error(f"Failed to load models: {e}")
            raise
        
        return models
    
    def extract_face(self, image_path: Union[str, Path]) -> FaceData:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        boxes, probs, landmarks = self.models['mtcnn'].detect(img, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            raise ValueError(f"No face detected in {image_path}")
        
        face_idx = 0
        if len(boxes) > 1:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            face_idx = np.argmax(areas)
        
        box = boxes[face_idx]
        prob = probs[face_idx]
        landmark = landmarks[face_idx] if landmarks is not None else None
        
        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_np.shape[1], x2), min(img_np.shape[0], y2)
        
        face_img = img_np[y1:y2, x1:x2]
        
        aligned_face = self.preprocessor.preprocess(face_img)
        
        quality_metrics = self.quality_assessor.assess_quality(aligned_face)
        
        embedding = self._get_embedding(aligned_face, image_path)
        
        landmark_dict = None
        if landmark is not None:
            landmark_dict = {
                'left_eye': landmark[0],
                'right_eye': landmark[1],
                'nose': landmark[2],
                'mouth_left': landmark[3],
                'mouth_right': landmark[4]
            }
        
        return FaceData(
            aligned_face=aligned_face,
            embedding=embedding,
            confidence=float(prob),
            bbox=[x1, y1, x2, y2],
            landmarks=landmark_dict,
            quality_score=quality_metrics['overall']
        )
    
    def _get_embedding(self, face: np.ndarray, image_path: Optional[Path] = None) -> np.ndarray:
        if self.cache and image_path:
            cached = self.cache.get(str(image_path), self.model_type.value)
            if cached is not None:
                return cached
        
        face_tensor = transforms.ToTensor()(face).unsqueeze(0).to(self.device)
        
        embeddings = []
        
        with torch.no_grad():
            if self.model_type == ModelType.ENSEMBLE:
                if 'facenet' in self.models:
                    emb = self.models['facenet'](face_tensor).cpu().numpy()[0]
                    embeddings.append(emb / np.linalg.norm(emb))
                
                if 'vggface' in self.models:
                    emb = self.models['vggface'](face_tensor).cpu().numpy()[0]
                    embeddings.append(emb / np.linalg.norm(emb))
                
                embedding = np.mean(embeddings, axis=0)
            else:
                model_name = self.model_type.value
                if model_name in self.models:
                    embedding = self.models[model_name](face_tensor).cpu().numpy()[0]
                else:
                    embedding = self.models['facenet'](face_tensor).cpu().numpy()[0]
                
                embedding = embedding / np.linalg.norm(embedding)
        
        if self.cache and image_path:
            self.cache.set(str(image_path), self.model_type.value, embedding)
        
        return embedding
    
    def calculate_similarity(self,
                           embedding1: np.ndarray,
                           embedding2: np.ndarray,
                           metric: DistanceMetric = DistanceMetric.COSINE) -> float:
        
        if metric == DistanceMetric.COSINE:
            return float(np.dot(embedding1, embedding2))
        elif metric == DistanceMetric.EUCLIDEAN:
            return float(1.0 / (1.0 + euclidean(embedding1, embedding2)))
        elif metric == DistanceMetric.MANHATTAN:
            return float(1.0 / (1.0 + np.sum(np.abs(embedding1 - embedding2))))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def compare_faces(self,
                     image1_path: Union[str, Path],
                     image2_path: Union[str, Path],
                     threshold: Optional[float] = None,
                     metric: DistanceMetric = DistanceMetric.COSINE) -> SimilarityResult:
        
        try:
            face1 = self.extract_face(image1_path)
            face2 = self.extract_face(image2_path)
        except Exception as e:
            logger.error(f"Face extraction failed: {e}")
            raise
        
        similarity = self.calculate_similarity(face1.embedding, face2.embedding, metric)
        
        if threshold is None:
            threshold = self.thresholds[self.model_type]
        
        confidence = (face1.confidence + face2.confidence) / 2.0
        
        quality_weight = (face1.quality_score + face2.quality_score) / 2.0
        adjusted_similarity = similarity * (0.7 + 0.3 * quality_weight)
        
        is_same = adjusted_similarity >= threshold
        
        if 0.6 <= adjusted_similarity < threshold:
            if face1.quality_score < 0.5 or face2.quality_score < 0.5:
                is_same = True
                confidence *= 0.8
        
        details = {
            'raw_similarity': similarity,
            'adjusted_similarity': adjusted_similarity,
            'quality_adjustment': quality_weight,
            'face1_detection_confidence': face1.confidence,
            'face2_detection_confidence': face2.confidence
        }
        
        return SimilarityResult(
            similarity_score=adjusted_similarity,
            is_same_person=is_same,
            confidence=confidence,
            distance_metric=metric.value,
            model_used=self.model_type.value,
            threshold_used=threshold,
            face1_quality=face1.quality_score,
            face2_quality=face2.quality_score,
            details=details
        )
    
    def compare_batch(self,
                     reference_image: Union[str, Path],
                     comparison_images: List[Union[str, Path]],
                     threshold: Optional[float] = None,
                     max_workers: int = 4) -> List[SimilarityResult]:
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for img_path in comparison_images:
                future = executor.submit(
                    self.compare_faces,
                    reference_image,
                    img_path,
                    threshold
                )
                futures.append(future)
            
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Comparison failed: {e}")
                    results.append(None)
        
        return results
    
    def find_similar_faces(self,
                          query_image: Union[str, Path],
                          database_images: List[Union[str, Path]],
                          top_k: int = 5,
                          threshold: float = 0.5) -> List[Tuple[str, SimilarityResult]]:
        
        results = self.compare_batch(query_image, database_images)
        
        scored_results = []
        for img_path, result in zip(database_images, results):
            if result and result.similarity_score >= threshold:
                scored_results.append((str(img_path), result))
        
        scored_results.sort(key=lambda x: x[1].similarity_score, reverse=True)
        
        return scored_results[:top_k]
    
    def visualize_comparison(self,
                            face1: FaceData,
                            face2: FaceData,
                            result: SimilarityResult) -> np.ndarray:
        
        face1_img = cv2.resize(face1.aligned_face, (160, 160))
        face2_img = cv2.resize(face2.aligned_face, (160, 160))
        
        diff = cv2.absdiff(face1_img, face2_img)
        
        heatmap = cv2.applyColorMap(diff.mean(axis=2).astype(np.uint8), cv2.COLORMAP_JET)
        
        comparison = np.hstack([face1_img, face2_img, diff, heatmap])
        
        h, w = comparison.shape[:2]
        info_panel = np.ones((80, w, 3), dtype=np.uint8) * 255
        
        texts = [
            f"Similarity: {result.similarity_score:.2%}",
            f"Same Person: {result.is_same_person}",
            f"Confidence: {result.confidence:.2%}",
            f"Model: {result.model_used}"
        ]
        
        y_offset = 20
        for text in texts:
            cv2.putText(info_panel, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 20
        
        final_image = np.vstack([comparison, info_panel])
        
        return final_image


def create_html_report(results: List[SimilarityResult], 
                       output_path: str = "similarity_report.html"):
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Similarity Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .result {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; }}
            .same {{ background-color: #d4edda; }}
            .different {{ background-color: #f8d7da; }}
            .metrics {{ margin: 10px 0; }}
            .metric {{ display: inline-block; margin: 0 15px; }}
        </style>
    </head>
    <body>
        <h1>Face Similarity Analysis Report</h1>
        {content}
    </body>
    </html>
    """
    
    content = ""
    for i, result in enumerate(results):
        class_name = "same" if result.is_same_person else "different"
        content += f"""
        <div class="result {class_name}">
            <h3>Comparison {i+1}</h3>
            <div class="metrics">
                <span class="metric"><b>Similarity:</b> {result.similarity_score:.2%}</span>
                <span class="metric"><b>Same Person:</b> {result.is_same_person}</span>
                <span class="metric"><b>Confidence:</b> {result.confidence:.2%}</span>
                <span class="metric"><b>Model:</b> {result.model_used}</span>
            </div>
        </div>
        """
    
    with open(output_path, 'w') as f:
        f.write(html_template.format(content=content))
    
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    face_sim = EnhancedFaceSimilarity(model_type=ModelType.ENSEMBLE)
    
    import os
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    
    test_pairs = [
        ("gettyimages-1165177751-2000-d105ca4fa12c49efae54bba6cac13822.jpg", "images.jpg"),
        ("Margot_Robbie_Mug.jpg", "0ed87b58-b9a2-4ed4-bc5e-69ff20f3c313_f750x750.jpg"),
    ]
    
    for img1, img2 in test_pairs:
        try:
            img1_path = os.path.join(desktop_path, img1)
            img2_path = os.path.join(desktop_path, img2)
            
            result = face_sim.compare_faces(img1_path, img2_path)
            
            print(f"\nComparing {img1} vs {img2}:")
            print(f"  Similarity: {result.similarity_score:.2%}")
            print(f"  Same Person: {result.is_same_person}")
            print(f"  Confidence: {result.confidence:.2%}")
            
        except Exception as e:
            print(f"Error comparing {img1} and {img2}: {e}")