# No AI used in this source code, authored by HG on 6/3/26 and edited by HG on 6/10/26
# UI enhanced with Claude AI (claude.ai)
# V1: Risky vs Safe Form Classification
# V2: Pose Classification (Downdog vs Plank) using MobileNetV2
# V3: Risk Assessment Layer added - needs enhancement

#CLAUDE WAS USED 6/10 TO ENHANCE APP UI, SOURCE : https://claude.ai/chat/d74f674d-e8e4-46d6-a7c0-d5ac56dbe1da

import gradio as gr
import torch
from torchvision import models, transforms

# -------------------------
# Pose Model V2 which shows the Downdog vs Plank output
# -------------------------
pose_model = models.mobilenet_v2(pretrained=False)
pose_model.classifier[1] = torch.nn.Linear(pose_model.last_channel, 2)
pose_model.load_state_dict(
    torch.load("models/pose_class_deep_learning.pth", map_location=torch.device("cpu"))
)
pose_model.eval()


# Risk Model v3 to assess and return safe versus risky forms
# -------------------------
risk_model = models.mobilenet_v2(pretrained=False)
risk_model.classifier[1] = torch.nn.Linear(risk_model.last_channel, 2)
risk_model.load_state_dict(
    torch.load("models/pilates_model_with_aug.pth", map_location=torch.device("cpu"))
)
risk_model.eval()

# Label here
classes = ["Downdog", "Plank"]
risk_classes = ["Risky Form", "Safe Form"]

# -------------------------
# Transforming
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
#Prediction Function
# -------------------------
def predict(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Pose classification
        outputs = pose_model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        label = classes[predicted.item()]
        confidence_percent = round(confidence.item() * 100, 2)

        downdog_probability = round(probabilities[0].item() * 100, 2)
        plank_probability = round(probabilities[1].item() * 100, 2)

        # Risk assessment for only planks since no bad downdog
        ## came across issue: risk model undertrained due to small dataset size and class imbalance
        # so model output probabilities were nearly identical across safe and risky samples 
        #...so it failed to learn meaningful distinguishing features. Flagged for V4 retraining with larger labeled dataset.
        risk_result = "Not Evaluated"
        if label == "Plank":
            risk_result = "IMPORTANT NOTE: Risk model needs more training data for accurate form assessment - coming in V4"

    # -------------------------
    # Build styled HTML output card
    # -------------------------
    # Color-code confidence: green ≥80%, yellow 60–79%, red <60%
    confidence_color = "#00E5A0" if confidence_percent >= 80 else "#FFD166" if confidence_percent >= 60 else "#FF6B6B"

    # Confidence interpretation line
    if confidence_percent >= 80:
        confidence_interp = "Strong detection — high model certainty."
        interp_color = "#00E5A0"
    elif confidence_percent >= 60:
        confidence_interp = "Moderate confidence — ensure your full body is visible in frame."
        interp_color = "#FFD166"
    else:
        confidence_interp = "Low confidence — try a clearer angle, better lighting, or step further back."
        interp_color = "#FF6B6B"

    pose_icon = "🐾" if label == "Downdog" else "💪"

    # ── Hardcoded form cues by pose (UI only, no inference) ──
    if label == "Plank":
        form_cues = [
            "Shoulders stacked directly over wrists, elbows soft — not locked",
            "Hips level with shoulders: no sagging lower back, no raised seat",
            "Core and glutes braced — imagine pulling your navel toward your spine",
            "Gaze slightly forward of hands to keep a neutral cervical spine",
        ]
    else:  # Downdog
        form_cues = [
            "Hinge deep at the hips — think of creating an inverted V, not a flat back",
            "Press heels toward the mat while keeping a micro-bend in the knees",
            "Lengthen through the entire spine: reach the tailbone up and back",
            "Broaden across the shoulder blades and rotate upper arms outward",
        ]

    cues_items = "".join(
        f'<li class="cue-item"><span class="cue-dot"></span>{cue}</li>'
        for cue in form_cues
    )

    # ── Risk section: roadmap note styling for Plank, neutral for Downdog ──
    if label == "Plank":
        risk_html = """
        <div class="risk-banner roadmap">
            <span class="risk-icon">🗺️</span>
            <div>
                <div class="risk-label">Risk Assessment</div>
                <div class="risk-roadmap-text">
                    Coming in V4 — expanding to real-time multi-pose risk analysis
                    and dynamic form assessment with a larger labeled dataset.
                </div>
            </div>
        </div>
        """
    else:
        risk_html = """
        <div class="risk-banner safe">
            <span class="risk-icon">✅</span>
            <div>
                <div class="risk-label">Risk Assessment</div>
                <div class="risk-text">Not evaluated for Downdog pose</div>
            </div>
        </div>
        """

    html_output = f"""
    <div class="result-card">

        <!-- Pose header -->
        <div class="pose-header">
            <span class="pose-icon">{pose_icon}</span>
            <div>
                <div class="pose-eyebrow">Detected Pose</div>
                <div class="pose-name">{label.upper()}</div>
            </div>
            <div class="confidence-badge" style="background: {confidence_color}22; border-color: {confidence_color}; color: {confidence_color};">
                {confidence_percent}%<br><span class="badge-sub">confidence</span>
            </div>
        </div>

        <!-- Confidence meter -->
        <div class="meter-section">
            <div class="meter-label-row">
                <span class="meter-label">MODEL CONFIDENCE</span>
                <span class="meter-val" style="color: {confidence_color};">{confidence_percent}%</span>
            </div>
            <div class="meter-track">
                <div class="meter-fill" style="width: {confidence_percent}%; background: {confidence_color};"></div>
            </div>
            <div class="confidence-interp" style="color: {interp_color};">{confidence_interp}</div>
        </div>

        <!-- Probability breakdown -->
        <div class="probs-grid">
            <div class="prob-card">
                <div class="prob-eyebrow">🐾 Downdog</div>
                <div class="prob-value">{downdog_probability}<span class="prob-pct">%</span></div>
                <div class="prob-track">
                    <div class="prob-fill dd" style="width: {downdog_probability}%;"></div>
                </div>
            </div>
            <div class="prob-card">
                <div class="prob-eyebrow">💪 Plank</div>
                <div class="prob-value">{plank_probability}<span class="prob-pct">%</span></div>
                <div class="prob-track">
                    <div class="prob-fill pl" style="width: {plank_probability}%;"></div>
                </div>
            </div>
        </div>

        <!-- Form cues -->
        <div class="cues-section">
            <div class="cues-header">
                <span class="cues-icon">📐</span>
                <span class="cues-title">Form Cues to Watch</span>
            </div>
            <ul class="cues-list">{cues_items}</ul>
        </div>

        <!-- Risk assessment -->
        {risk_html}

    </div>
    """
    return html_output


# -------------------------
# Custom CSS
# -------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0F0F1A !important;
    font-family: 'Inter', sans-serif !important;
    color: #F5F0E8 !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 0 16px 48px !important;
}

/* ── Hero header ── */
.hero-wrap {
    text-align: center;
    padding: 48px 0 36px;
    border-bottom: 1px solid rgba(0,229,160,0.15);
    margin-bottom: 36px;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #00E5A0;
    margin-bottom: 12px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(28px, 5vw, 44px);
    font-weight: 700;
    color: #F5F0E8;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 12px;
}
.hero-title span { color: #00E5A0; }
.hero-subtitle {
    font-size: 15px;
    color: #8A8A9E;
    max-width: 440px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Upload zone ── */
.upload-section {
    background: #16162A;
    border: 1.5px dashed rgba(0,229,160,0.35);
    border-radius: 16px;
    padding: 8px;
    margin-bottom: 24px;
    transition: border-color 0.2s;
}
.upload-section:hover { border-color: rgba(0,229,160,0.65); }

.upload-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00E5A0;
    padding: 10px 14px 2px;
}

/* Gradio image component overrides */
.upload-section .wrap { border: none !important; background: transparent !important; }
.upload-section .image-container { border-radius: 10px !important; overflow: hidden; }

/* ── Analyze button ── */
.analyze-btn {
    width: 100% !important;
    background: #00E5A0 !important;
    color: #0F0F1A !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px !important;
    cursor: pointer !important;
    transition: background 0.15s, transform 0.1s !important;
    margin-bottom: 28px !important;
}
.analyze-btn:hover { background: #00CCB5 !important; transform: translateY(-1px) !important; }
.analyze-btn:active { transform: translateY(0) !important; }

/* ── Results section label ── */
.results-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8A8A9E;
    margin-bottom: 12px;
}

/* ── Result card (rendered inside gr.HTML) ── */
.result-card {
    background: #16162A;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid rgba(255,255,255,0.07);
}

.pose-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 28px;
}
.pose-icon { font-size: 36px; line-height: 1; }
.pose-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8A8A9E;
    margin-bottom: 4px;
}
.pose-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    color: #F5F0E8;
    letter-spacing: -0.01em;
}
.confidence-badge {
    margin-left: auto;
    border: 1.5px solid;
    border-radius: 10px;
    padding: 10px 16px;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.2;
    min-width: 84px;
}
.badge-sub {
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.8;
}

/* Confidence meter */
.meter-section { margin-bottom: 24px; }
.meter-label-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
}
.meter-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    color: #8A8A9E;
}
.meter-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 13px;
    font-weight: 700;
}
.meter-track {
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    overflow: hidden;
}
.meter-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}

/* Probability grid */
.probs-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 24px;
}
.prob-card {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.06);
}
.prob-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8A8A9E;
    margin-bottom: 6px;
}
.prob-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #F5F0E8;
    line-height: 1;
    margin-bottom: 10px;
}
.prob-pct {
    font-size: 14px;
    font-weight: 500;
    color: #8A8A9E;
}
.prob-track {
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 99px;
}
.prob-fill.dd { background: #00E5A0; }
.prob-fill.pl { background: #A78BFA; }

/* Confidence interpretation line */
.confidence-interp {
    font-size: 12px;
    font-style: italic;
    margin-top: 8px;
    line-height: 1.5;
    opacity: 0.9;
}

/* Form cues section */
.cues-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.cues-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}
.cues-icon { font-size: 15px; }
.cues-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #F5F0E8;
}
.cues-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.cue-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    font-size: 13px;
    color: #C8C4BE;
    line-height: 1.55;
}
.cue-dot {
    flex-shrink: 0;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #00E5A0;
    margin-top: 6px;
}

/* Risk banner */
.risk-banner {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    border-radius: 12px;
    padding: 14px 16px;
}
.risk-banner.safe {
    background: rgba(0,229,160,0.07);
    border: 1px solid rgba(0,229,160,0.2);
}
.risk-banner.roadmap {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
}
.risk-icon { font-size: 16px; margin-top: 2px; flex-shrink: 0; }
.risk-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #8A8A9E;
    margin-bottom: 4px;
}
.risk-text {
    font-size: 13px;
    color: #F5F0E8;
    line-height: 1.5;
}
/* Roadmap note: muted, small, not bold */
.risk-roadmap-text {
    font-size: 11.5px;
    color: #5A5A72;
    line-height: 1.55;
    font-style: italic;
}

/* ── Footer ── */
.footer-wrap {
    text-align: center;
    margin-top: 40px;
    padding-top: 24px;
    border-top: 1px solid rgba(255,255,255,0.07);
}
.footer-text {
    font-size: 12px;
    color: #3D3D56;
    letter-spacing: 0.04em;
}
.footer-text a { color: #00E5A0; text-decoration: none; }

/* ── Gradio chrome cleanup ── */
footer { display: none !important; }
.gr-prose h1, .gr-prose h2, .gr-prose p { display: none !important; }
#component-0 { gap: 0 !important; }
.contain { background: transparent !important; }
.panel-header { display: none !important; }
"""

# -------------------------
# Launch — gr.Blocks() layout
# -------------------------
with gr.Blocks(css=custom_css, title="PilatesAI — Form Detector") as demo:

    # Hero
    gr.HTML("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">AI-Powered · MobileNetV2 · Duke AIPI</div>
        <h1 class="hero-title">Pilates Form<br><span>Detector</span></h1>
        <p class="hero-subtitle">
            Upload a photo of your Downdog or Plank and get instant
            pose classification and form feedback.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="upload-label">Upload Your Photo</div>')
            image_input = gr.Image(
                type="pil",
                elem_classes=["upload-section"],
                show_label=False,
                height=300,
            )
            analyze_btn = gr.Button(
                "⚡ Analyze Form",
                elem_classes=["analyze-btn"],
                variant="primary",
            )

        with gr.Column(scale=1):
            gr.HTML('<div class="results-label">Analysis Results</div>')
            output_html = gr.HTML(
                value="""
                <div class="result-card" style="min-height:260px; display:flex; align-items:center; justify-content:center; flex-direction:column; gap:10px;">
                    <div style="font-size:40px; opacity:0.3;">🧘</div>
                    <div style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#3D3D56; text-transform:uppercase; letter-spacing:0.14em;">
                        Awaiting upload
                    </div>
                </div>
                """
            )

    analyze_btn.click(fn=predict, inputs=image_input, outputs=output_html)

    # Footer
    gr.HTML("""
    <div class="footer-wrap">
        <p class="footer-text">
            V3 · Duke University AIPI &nbsp;·&nbsp; UI enhanced with <a href="https://claude.ai" target="_blank">Claude AI</a>
        </p>
    </div>
    """)

demo.launch(share=True)