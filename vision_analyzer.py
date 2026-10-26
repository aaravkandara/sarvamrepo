import os
import torch
import torchvision
import skimage.io
import torchxrayvision as xrv

_model = None

def get_xray_model():
    """Lazy-load the DenseNet-121 chest X-ray model."""
    global _model
    if _model is None:
        _model = xrv.models.DenseNet(weights="densenet121-res224-all")
        _model.eval()
    return _model

def analyze_xray(image_path: str, top_k: int = 5, threshold: float = 0.35) -> dict:
    """
    Analyze a chest X-ray image and return detected pathologies with probabilities.
    
    Args:
        image_path: Path to the X-ray image (PNG, JPG, etc.)
        top_k: Maximum number of top findings to highlight
        threshold: Minimum confidence score to flag a pathology as significant
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"X-ray image not found: {image_path}")

    # Load and normalize image to [-1024, 1024]
    img = skimage.io.imread(image_path)
    img = xrv.datasets.normalize(img, 255)

    # Convert to single channel (grayscale) if needed
    if len(img.shape) > 2:
        img = img.mean(2)
    img = img[None, ...]

    # Resize and crop to 224x224
    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224)
    ])
    img = transform(img)
    img = torch.from_numpy(img).unsqueeze(0)

    model = get_xray_model()
    with torch.no_grad():
        preds = model(img)

    all_scores = {
        pathology: round(float(prob), 4)
        for pathology, prob in zip(model.pathologies, preds[0])
    }

    # Sort by descending probability
    sorted_findings = sorted(all_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Flag significant findings above threshold
    significant_findings = [
        {"pathology": name, "score": score}
        for name, score in sorted_findings
        if score >= threshold
    ]

    return {
        "image_path": image_path,
        "top_findings": [{"pathology": k, "score": v} for k, v in sorted_findings[:top_k]],
        "significant_findings": significant_findings,
        "all_scores": all_scores
    }

if __name__ == "__main__":
    import json
    results = analyze_xray("sample_xray.png")
    print(json.dumps(results, indent=2))
