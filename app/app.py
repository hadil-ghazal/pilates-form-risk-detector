import gradio as gr
import torch

from torchvision import models, transforms
from PIL import Image

#Loadingmodel
model = models.mobilenet_v2(pretrained=False)

model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(
    torch.load(
        "models/pilates_model_with_aug.pth",
        map_location=torch.device("cpu")
    ))
model.eval()

classes = ["Risky Form", "Safe Form"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict(image):

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Pilates Posture Risk Detector")

##demo.launch()
demo.launch(share=True)