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

   ## with torch.no_grad():
    ##    outputs = model(image)
    ##    _, predicted = torch.max(outputs, 1)

   ## return classes[predicted.item()]

## V2 ENHANCEMENT TO ADD CONFIDENCE LEVEL TO  DETECTOR OUTPUT
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        label = classes[predicted.item()]
        confidence_percent = round(confidence.item() * 100, 2)
    #return f"{label} ({confidence_percent}% model confidence)"
    #V3: adding better descriptions in the output for UI enhancement
        safe_probability = round(probabilities[1].item() * 100, 2)
        risky_probability = round(probabilities[0].item() * 100, 2)

    if label == "Risky Form":

        recommendation = "Recommended correction: maintain a neutral spine and stable shoulder alignment."

    else:

        recommendation = "Form appears stable. Continue maintaining alignment."

    return f"""
        Prediction: {label}
        Safe Form Probability: {safe_probability}%
        Risky Form Probability: {risky_probability}%
        {recommendation}
"""
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    #outputs="json",
    title="Pilates Posture Risk Detector")

##demo.launch()
demo.launch(share=True)