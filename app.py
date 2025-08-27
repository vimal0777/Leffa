# app.py — Vimal AI Outfit App (minimal UI, robust validation, PWA)
# - Virtual Try-On (no pose transfer)
# - Upload/Webcam + URL load for Person, Upper, Lower, Dress
# - "Both (Upper + Lower)" = sequential passes (upper then lower)
# - Advanced Options: ONLY "Garment Mode" (Model Type locked to DressCode)
# - Debug accordion hidden
# - Equal-height first row & responsive second row
# - Gradio branding/Settings hidden via CSS
# - Strong input validation + error surfacing
# - PWA enabled via demo.launch(pwa=True)

import os
import tempfile
from io import BytesIO
import numpy as np
import requests
from PIL import Image, ImageFile
from huggingface_hub import snapshot_download

from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

import gradio as gr

# Make PIL tolerant of slightly truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------------------
# Download checkpoints (idempotent)
# ---------------------------------------------------------------------
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

# ---------------------------------------------------------------------
# Core Predictor (Virtual Try-On only)
# ---------------------------------------------------------------------
class LeffaPredictor(object):
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )
        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )
        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )
        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_dc)

    def leffa_predict_vt(
        self,
        src_image_path: str,
        ref_image_path: str,
        *,
        vt_model_type: str = "dress_code",   # locked default (hidden in UI)
        vt_garment_type: str = "upper_body",
        # hidden defaults
        ref_acceleration: bool = False,
        step: int = 30,
        scale: float = 2.5,
        seed: int = 42,
        repaint: bool = False,
    ):
        if not src_image_path:
            raise gr.Error("Please upload/capture a Person Image or load it from URL.")
        if not ref_image_path:
            raise gr.Error("Please upload/capture the selected garment image or load it from URL.")

        try:
            src_image = Image.open(src_image_path).convert("RGB")
            ref_image = Image.open(ref_image_path).convert("RGB")
        except Exception as e:
            raise gr.Error(f"Could not open one of the images. {type(e).__name__}: {e}")

        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        # Mask
        try:
            model_parse, _ = self.parsing(src_image.resize((384, 512)))
            keypoints = self.openpose(src_image.resize((384, 512)))
            if vt_model_type == "viton_hd":
                mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
            else:
                mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
            mask = mask.resize((768, 1024))
        except Exception as e:
            raise gr.Error(f"Failed while creating the agnostic mask. {type(e).__name__}: {e}")

        # DensePose
        try:
            src_image_array = np.array(src_image)
            if vt_model_type == "viton_hd":
                seg = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
                densepose = Image.fromarray(seg)
            else:
                iuv = self.densepose_predictor.predict_iuv(src_image_array)
                seg = iuv[:, :, 0:1]
                seg = np.concatenate([seg] * 3, axis=-1)
                densepose = Image.fromarray(seg)
        except Exception as e:
            raise gr.Error(f"DensePose failed. {type(e).__name__}: {e}")

        # Inference
        try:
            transform = LeffaTransform()
            data = {
                "src_image": [src_image],
                "ref_image": [ref_image],
                "mask": [mask],
                "densepose": [densepose],
            }
            data = transform(data)
            inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
            output = inference(
                data,
                ref_acceleration=ref_acceleration,
                num_inference_steps=step,
                guidance_scale=scale,
                seed=seed,
                repaint=repaint,
            )
            gen_image = output["generated_image"][0]
        except Exception as e:
            raise gr.Error(f"Generation failed. {type(e).__name__}: {e}")

        return np.array(gen_image)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _path_exists(p): 
    return bool(p) and os.path.exists(p)

def _ensure_image_present(path, label):
    if not path:
        raise gr.Error(f"Please provide the {label} (upload/capture or URL).")
    if not os.path.exists(path):
        raise gr.Error(f"{label} not found on disk: {path}")
    try:
        with Image.open(path) as im:
            im.verify()
    except Exception:
        raise gr.Error(f"{label} is not a valid image: {path}")

def _save_np_image_to_tmp(np_img: np.ndarray) -> str:
    pil_img = Image.fromarray(np_img)
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    pil_img.save(tmp_path)
    return tmp_path

def _download_image_to_tmp(url: str) -> str:
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise gr.Error("Please provide a valid http(s) image URL.")
    try:
        headers = {"User-Agent": "VimalAI-OutfitApp/1.0"}
        resp = requests.get(url, timeout=25, stream=True, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        raw = resp.content
        img = Image.open(BytesIO(raw)).convert("RGB")
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp_path)
        return tmp_path
    except Exception as e:
        raise gr.Error(f"Failed to load image from URL. {type(e).__name__}: {e}")

# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    leffa_predictor = LeffaPredictor()

    custom_css = """
    :root{
      --brand:#2563eb; --brand-2:#3b82f6; --ink:#1f2937; --muted:#6b7280;
      --card:#ffffff; --bg:#f7f8fb; --shadow:0 8px 24px rgba(0,0,0,.06); --radius:14px;
      --garment-col-min: 340px;
    }
    body,.gradio-container{background:var(--bg);}
    footer, .gradio-container footer, .gradio-container .footer { display:none !important; }
    a[href*="gradio.app"], a[aria-label="Use via API"], a[aria-label="Built with Gradio"] { display:none !important; }
    button[aria-label="Settings"] { display:none !important; }

    .hero{text-align:center;max-width:980px;margin:28px auto 10px auto}
    .hero h1{font-size:38px;font-weight:800;color:var(--brand)}
    .hero p{color:var(--muted)}

    .grid-row1{
      max-width:1280px;margin:18px auto;display:grid;
      grid-template-columns:repeat(3,minmax(0,1fr));gap:18px;align-items:stretch;
    }
    .grid-row1 > .card{height:100%;display:flex;flex-direction:column;}
    .grid-row1 .card .gradio-image{flex:1 1 auto;}
    .grid-row1 .card .gradio-image, .grid-row1 .card .gradio-image > div,
    .grid-row1 .card .image-container, .grid-row1 .card .image-preview, .grid-row1 .card .wrap{min-height:0;}
    @media (min-width:1024px){ .grid-row1 > .card{ min-height:560px; } }
    @media (max-width:900px){ .grid-row1{ grid-template-columns:1fr; } }

    .grid-3{
      max-width:1280px;margin:6px auto 28px;display:grid;
      grid-template-columns:repeat(auto-fit, minmax(var(--garment-col-min), 1fr));
      gap:18px;align-items:stretch;
    }
    .grid-3 > .card{height:100%;display:flex;flex-direction:column;}
    .grid-3 .card .gradio-image{flex:1 1 auto;}
    .grid-3 .card .gradio-image, .grid-3 .card .gradio-image > div,
    .grid-3 .card .image-container, .grid-3 .card .image-preview{min-height:0;}

    .card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px}
    .card h3{margin:0 0 10px 0;color:var(--ink)}
    .btn{margin-top:12px;background:var(--brand);color:#fff;border:none;border-radius:12px;padding:12px 16px;font-weight:800;cursor:pointer;box-shadow:var(--shadow)}
    .btn-secondary{margin-top:8px;background:#fff;color:var(--brand);border:1px solid var(--brand);border-radius:10px;padding:8px 12px;font-weight:700;cursor:pointer}
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Vimal AI Outfit App") as demo:
        gr.HTML("""
        <section class="hero">
          <h1>Vimal AI Labs</h1>
          <p>Upload, capture, or paste an image URL for person and garments. Supports applying both Upper and Lower in one go.</p>
        </section>
        """)

        # -------- FIRST ROW --------
        with gr.Row(elem_classes="grid-row1"):
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Person Image")
                with gr.Column(elem_classes="wrap"):
                    person_image = gr.Image(sources=["upload", "webcam"], type="filepath", label="", height=260)
                person_url = gr.Textbox(label="Or paste Person Image URL", placeholder="https://example.com/photo.jpg")
                load_person_btn = gr.Button("Load Person from URL", elem_classes="btn-secondary")

            with gr.Column(elem_classes="card"):
                gr.Markdown("### Generated Image")
                with gr.Column(elem_classes="wrap"):
                    output_image = gr.Image(label="", height=260)
                generate_btn = gr.Button("Generate", elem_classes="btn")

            with gr.Column(elem_classes="card"):
                gr.Markdown("### ⚙️ Advanced Options")
                vt_garment_type = gr.Radio(
                    label="Garment Mode",
                    choices=[("Upper", "upper_body"),
                             ("Lower", "lower_body"),
                             ("Dress", "dresses"),
                             ("Both (Upper + Lower)", "both")],
                    value="upper_body",
                )

        # -------- SECOND ROW --------
        with gr.Row(elem_classes="grid-3"):
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Upper Garment")
                garment_upper = gr.Image(sources=["upload", "webcam"], type="filepath",
                                         label="Upload/Capture Upper Garment", height=240)
                upper_url = gr.Textbox(label="Or paste Upper Garment URL", placeholder="https://example.com/top.png")
                load_upper_btn = gr.Button("Load Upper from URL", elem_classes="btn-secondary")

            with gr.Column(elem_classes="card"):
                gr.Markdown("### Lower Garment")
                garment_lower = gr.Image(sources=["upload", "webcam"], type="filepath",
                                         label="Upload/Capture Lower Garment", height=240)
                lower_url = gr.Textbox(label="Or paste Lower Garment URL", placeholder="https://example.com/bottom.jpg")
                load_lower_btn = gr.Button("Load Lower from URL", elem_classes="btn-secondary")

            with gr.Column(elem_classes="card"):
                gr.Markdown("### Dress")
                garment_dress = gr.Image(sources=["upload", "webcam"], type="filepath",
                                         label="Upload/Capture Dress", height=240)
                dress_url = gr.Textbox(label="Or paste Dress URL", placeholder="https://example.com/dress.webp")
                load_dress_btn = gr.Button("Load Dress from URL", elem_classes="btn-secondary")

        # ---------- URL Loaders ----------
        def load_url_to_image(url: str):
            return _download_image_to_tmp(url)

        load_person_btn.click(load_url_to_image, inputs=[person_url], outputs=[person_image])
        load_upper_btn.click(load_url_to_image, inputs=[upper_url], outputs=[garment_upper])
        load_lower_btn.click(load_url_to_image, inputs=[lower_url], outputs=[garment_lower])
        load_dress_btn.click(load_url_to_image, inputs=[dress_url], outputs=[garment_dress])

        # ---------- Generate Handler ----------
        HIDDEN_MODEL_TYPE = "dress_code"
        HIDDEN_REF_ACCEL = False
        HIDDEN_STEP = 30
        HIDDEN_SCALE = 2.5
        HIDDEN_SEED = 42
        HIDDEN_REPAINT = False

        def run_vton(src_path, gmode, up_path, low_path, drs_path):
            # Validate required inputs based on garment mode
            _ensure_image_present(src_path, "Person Image")

            if gmode == "both":
                _ensure_image_present(up_path, "Upper garment")
                _ensure_image_present(low_path, "Lower garment")
            elif gmode == "upper_body":
                _ensure_image_present(up_path, "Upper garment")
            elif gmode == "lower_body":
                _ensure_image_present(low_path, "Lower garment")
            elif gmode == "dresses":
                _ensure_image_present(drs_path, "Dress")

            try:
                if gmode == "both":
                    # Pass 1: Upper
                    gen1 = leffa_predictor.leffa_predict_vt(
                        src_image_path=src_path,
                        ref_image_path=up_path,
                        vt_model_type=HIDDEN_MODEL_TYPE,
                        vt_garment_type="upper_body",
                        ref_acceleration=HIDDEN_REF_ACCEL,
                        step=HIDDEN_STEP,
                        scale=HIDDEN_SCALE,
                        seed=HIDDEN_SEED,
                        repaint=HIDDEN_REPAINT,
                    )
                    # Save intermediate
                    inter_path = _save_np_image_to_tmp(gen1)
                    # Pass 2: Lower
                    gen2 = leffa_predictor.leffa_predict_vt(
                        src_image_path=inter_path,
                        ref_image_path=low_path,
                        vt_model_type=HIDDEN_MODEL_TYPE,
                        vt_garment_type="lower_body",
                        ref_acceleration=HIDDEN_REF_ACCEL,
                        step=HIDDEN_STEP,
                        scale=HIDDEN_SCALE,
                        seed=HIDDEN_SEED,
                        repaint=HIDDEN_REPAINT,
                    )
                    return gen2

                elif gmode == "upper_body":
                    return leffa_predictor.leffa_predict_vt(
                        src_image_path=src_path,
                        ref_image_path=up_path,
                        vt_model_type=HIDDEN_MODEL_TYPE,
                        vt_garment_type="upper_body",
                        ref_acceleration=HIDDEN_REF_ACCEL,
                        step=HIDDEN_STEP,
                        scale=HIDDEN_SCALE,
                        seed=HIDDEN_SEED,
                        repaint=HIDDEN_REPAINT,
                    )

                elif gmode == "lower_body":
                    return leffa_predictor.leffa_predict_vt(
                        src_image_path=src_path,
                        ref_image_path=low_path,
                        vt_model_type=HIDDEN_MODEL_TYPE,
                        vt_garment_type="lower_body",
                        ref_acceleration=HIDDEN_REF_ACCEL,
                        step=HIDDEN_STEP,
                        scale=HIDDEN_SCALE,
                        seed=HIDDEN_SEED,
                        repaint=HIDDEN_REPAINT,
                    )

                elif gmode == "dresses":
                    return leffa_predictor.leffa_predict_vt(
                        src_image_path=src_path,
                        ref_image_path=drs_path,
                        vt_model_type=HIDDEN_MODEL_TYPE,
                        vt_garment_type="dresses",
                        ref_acceleration=HIDDEN_REF_ACCEL,
                        step=HIDDEN_STEP,
                        scale=HIDDEN_SCALE,
                        seed=HIDDEN_SEED,
                        repaint=HIDDEN_REPAINT,
                    )

                else:
                    raise gr.Error("Unknown garment mode. Please select Upper, Lower, Dress, or Both.")
            except gr.Error:
                raise
            except Exception as e:
                # Surface any unhandled backend error to the UI cleanly
                raise gr.Error(f"Unexpected error during generation: {type(e).__name__}: {e}")

        generate_btn.click(
            fn=run_vton,
            inputs=[person_image, vt_garment_type, garment_upper, garment_lower, garment_dress],
            outputs=[output_image]
        )

        demo.launch(share=True, server_port=7860, pwa=True)
