"""Quick demo: train on LFW and recognize a test face."""

from facerec_classical.pipeline import ClassicalFaceRecPipeline
from facerec_classical.config import Config
import cv2
import sys


def main():
    config = Config()
    pipeline = ClassicalFaceRecPipeline(config)

    # 1. Train on LFW (or pass a custom path)
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else str(config.data_dir / "lfw")
    print(f"Training on: {dataset_path}")
    metrics = pipeline.train(dataset_path=dataset_path)

    print("\n--- Training Metrics ---")
    print(f"  Samples:              {metrics['n_samples']}")
    print(f"  Classes:              {metrics['n_classes']}")
    print(f"  PCA components:       {metrics['n_pca_components']}")
    print(f"  LDA components:       {metrics['n_lda_components']}")
    print(f"  Cumulative variance:  {metrics['cumulative_variance']:.4f}")
    print(f"  Reconstruction error: {metrics['reconstruction_error']:.4f}")
    print(f"  Train accuracy (LOO): {metrics['train_accuracy']:.4f}")

    # 2. Recognize a test image (if provided)
    if len(sys.argv) > 2:
        test_path = sys.argv[2]
        print(f"\nRecognizing: {test_path}")
        image = cv2.imread(test_path)
        if image is None:
            print(f"Could not load image: {test_path}")
            return

        results = pipeline.recognize(image)
        for r in results:
            print(f"  {r.name} (SED: {r.distance:.2f}) at bbox {r.bbox}")
    else:
        print("\nTip: pass a test image path as 2nd arg to recognize a face")
        print("  python scripts/run_demo.py data/lfw path/to/test.jpg")


if __name__ == "__main__":
    main()
