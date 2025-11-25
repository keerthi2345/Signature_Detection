import torch, os
from dataset import SignatureDataset
from model import SignatureNet
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT,"weights","signature_model.pth")
DATA_DIR = os.path.join(ROOT,"data","processed")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SignatureNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
model.eval()

ds = SignatureDataset(root_dir=DATA_DIR, mode="eval")
loader = DataLoader(ds, batch_size=32, shuffle=False)

y_true = []
y_pred = []
with torch.no_grad():
    for imgs, labs in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        y_true.extend(labs.numpy().astype(int).tolist())
        y_pred.extend(preds.tolist())

print(classification_report(y_true, y_pred, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
