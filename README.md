# Face Similarity Comparison

A face similarity comparison system using OpenCV and multi-feature analysis to determine if two faces belong to the same person.

## Features

- **OpenCV Face Detection**: Haar Cascade-based face detection
- **Multi-Feature Analysis**: LBP, Gabor filters, HOG, color histograms
- **Dynamic Thresholding**: Adaptive similarity thresholds
- **JSON Results**: Detailed metrics storage

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_tests.py
```

Place image pairs in the project directory with naming pattern:
- `1 (1).jpg` and `1 (2).jpg` for pair 1
- `2 (1).jpg` and `2 (2).jpg` for pair 2

## Project Structure

```
├── run_tests.py                    # Main test runner
├── face_similarity_enhanced.py     # Advanced features
├── requirements.txt                # Dependencies
├── FaceNet + MTCNN with Landmark Alignment.ipynb
└── SimpleFaceAnalysis with Fallback Detection.ipynb
```

## Algorithm

1. **Face Detection**: Extract faces using OpenCV
2. **Feature Extraction**: Compute LBP, Gabor, HOG, and color features
3. **Similarity Calculation**: Weighted combination of features
4. **Dynamic Decision**: Apply pattern-specific thresholds

## Results

Similarity scores above 85% indicate likely same person. Results are saved to `test_results.json`.

## Dependencies

- `numpy`
- `pillow` 
- `opencv-python-headless`

## License

Educational and research purposes.