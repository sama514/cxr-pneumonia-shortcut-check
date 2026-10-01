"""Feature extraction: ResNet18 embeddings, a torch-free pixel embedder (for debugging), and the shortcut features."""
import numpy as np
from PIL import Image


def simple_features(paths):
    """Only size and brightness: width, height, aspect ratio, mean, std."""
    rows = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
            a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        rows.append([w, h, w / h, a.mean(), a.std()])
    return np.array(rows, dtype=np.float32)


def pixel_embed(paths, size=32):
    out = []
    for i, p in enumerate(paths, 1):
        with Image.open(p) as im:
            out.append(np.asarray(im.convert("L").resize((size, size)), dtype=np.float32).ravel() / 255.0)
        if i % 500 == 0 or i == len(paths):
            print(f"embedded {i}/{len(paths)}", flush=True)
    return np.stack(out)


def resnet_embed(paths, batch_size=32):
    """Frozen ImageNet ResNet18 -> 512-d vector per image. Needs torch + torchvision (downloads weights on first run)."""
    import torch
    from torchvision import models, transforms

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            imgs = []
            for p in chunk:
                with Image.open(p) as im:
                    imgs.append(tf(im.convert("RGB")))
            feats.append(model(torch.stack(imgs)).numpy())
            print(f"embedded {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(feats)


EMBEDDERS = {"resnet18": resnet_embed, "pixels": pixel_embed}
