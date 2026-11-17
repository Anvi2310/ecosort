"""MobileNetV2 inference adapter; never disguises ImageNet as waste training."""
from pathlib import Path
from PIL import Image
import numpy as np
from config import CATEGORIES, MODEL_DIR
CLASSIFIER_CLASSES=tuple(sorted(CATEGORIES, key=lambda x: x.lower().replace(' ','_')))

class WasteClassifier:
    def __init__(self, checkpoint=None):
        self.model=None; self.transform=None; self.status="demo_fallback"; self.error=None
        try:
            import torch
            from torchvision import models, transforms
            # Inference startup must be offline-safe. The explicitly invoked training
            # script downloads pretrained weights; this path loads only a saved checkpoint.
            self.model=models.mobilenet_v2(weights=None)
            checkpoint=Path(checkpoint or MODEL_DIR/"waste_mobilenet_v2.pth")
            if checkpoint.exists():
                state=torch.load(checkpoint,map_location="cpu",weights_only=True)
                self.model.classifier[1]=torch.nn.Linear(self.model.last_channel,len(CLASSIFIER_CLASSES))
                self.model.load_state_dict(state); self.status="fine_tuned_checkpoint"
            else:
                self.model=None; self.status="demo_fallback_no_waste_checkpoint"
            self.transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        except Exception as exc: self.error=str(exc)
    def predict(self,image):
        if self.model is None:
            # Image-content agnostic demo response: explicitly uninformative uniform confidence.
            return {"category":None,"confidence":None,"top3":[],"status":self.status,"message":"No fine-tuned waste checkpoint is available; demo fallback does not classify this image."}
        import torch
        img=Image.open(image).convert("RGB"); self.model.eval()
        with torch.no_grad(): probs=torch.softmax(self.model(self.transform(img).unsqueeze(0)),1)[0].numpy()
        order=np.argsort(probs)[::-1][:3]
        return {"category":CLASSIFIER_CLASSES[int(order[0])],"confidence":float(probs[order[0]]),"top3":[{"category":CLASSIFIER_CLASSES[int(i)],"confidence":float(probs[i])} for i in order],"status":self.status}
