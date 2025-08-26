# app.py — Vimal AI Outfit App (adds "Load from URL" for Person/Upper/Lower/Dress)
import os
import tempfile
import numpy as np
from io import BytesIO
import requests
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import gradio as gr

# ---------------------------------------------------------------------
# Download checkpoints (idempotent)
# ---------------------------------------------------------------------
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

# ---------------------------------------------------------------------
# Core Predictor (Virtual Try-On only)
# ---------------------------------------------------------------------
class LeffaPredictor(object):
    def __init__(self):
        # Pre/post utils
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

        # Virtual Try-On models
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
        ref_acceleration: bool = False,
        step: int = 30,
        scale: float = 2.5,
        seed: int = 42,
        vt_model_type: str = "dress_code",   # default to DressCode
        vt_garment_type: str = "upper_body",
        vt_repaint: bool = False
    ):
        """
        Virtual try-on inference wrapper for a SINGLE garment type (upper/lower/dress).
        """
        # Validate inputs
        if not src_image_path:
            raise gr.Error("Please upload/capture a Person Image or load it from URL.")
        if not ref_image_path:
            raise gr.Error("Please upload/capture the selected garment image or load it from URL.")

        src_image = Image.open(src_image_path)
        ref_image = Image.open(ref_image_path)
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        # ------- Mask (garment-agnostic) -------
        src_image_rgb = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image_rgb.resize((384, 512)))
        keypoints = self.openpose(src_image_rgb.resize((384, 512)))
        if vt_model_type == "viton_hd":
            mask = get_agnostic_mask_hd(model_parse, keypoints, vt_garment_type)
        else:
            mask = get_agnostic_mask_dc(model_parse, keypoints, vt_garment_type)
        mask = mask.resize((768, 1024))

        # ------- DensePose -------
        src_image_array = np.array(src_image)
        if vt_model_type == "viton_hd":
            src_image_seg_array = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
            densepose = Image.fromarray(src_image_seg_array)
        else:
            src_image_iuv_array = self.densepose_predictor.predict_iuv(src_image_array)
            src_image_seg_array = src_image_iuv_array[:, :, 0:1]
            src_image_seg_array = np.concatenate([src_image_seg_array] * 3, axis=-1)
            densepose = Image.fromarray(src_image_seg_array)

        # ------- Leffa inference -------
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
            repaint=vt_repaint,
        )
        gen_image = output["generated_image"][0]
        return np.array(gen_image), np.array(mask), np.array(densepose)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure(path, msg):
    if not path:
        raise gr.Error(msg)

def _save_np_image_to_tmp(np_img: np.ndarray) -> str:
    pil_img = Image.fromarray(np_img)
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    pil_img.save(tmp_path)
    return tmp_path

def _download_image_to_tmp(url: str) -> str:
    """
    Downloads an image from a URL, validates it's an image, and saves to a temp PNG file.
    Returns the temp file path.
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        raise gr.Error("Please provide a valid http(s) image URL.")
    try:
        headers = {"User-Agent": "VimalAI-OutfitApp/1.0"}
        resp = requests.get(url, timeout=20, stream=True, headers=headers, allow_redirects=True)
        resp.raise_for_status()
        # Validate content-type OR try PIL directly
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.content if hasattr(resp, "content") else resp.read()
        img = Image.open(BytesIO(raw)).convert("RGB")  # will throw if not an image
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        img.save(tmp_path)
        return tmp_path
    except Exception as e:
        raise gr.Error(f"Failed to load image from URL. {e}")

# ---------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    leffa_predictor = LeffaPredictor()

    # Example assets
    example_dir = "./ckpts/examples"
    person_images = list_dir(os.path.join(example_dir, "person1"))
    garment_examples = list_dir(os.path.join(example_dir, "garment"))

    # Styles
    custom_css = """
    :root{
      --brand:#2563eb; --brand-2:#3b82f6; --ink:#1f2937; --muted:#6b7280;
      --card:#ffffff; --bg:#f7f8fb; --shadow:0 8px 24px rgba(0,0,0,.06); --radius:14px;
    }
    body,.gradio-container{background:var(--bg);}
    .header{display:flex;align-items:center;justify-content:space-between;padding:16px 24px;background:var(--card);border-bottom:1px solid #eee;box-shadow:var(--shadow);}
    .brand{display:flex;align-items:center;gap:10px;color:var(--brand);font-weight:800}
    .brand .logo{width:28px;height:28px;border-radius:999px;background:var(--brand)}
    .nav{display:flex;gap:20px;color:var(--muted);font-weight:600}
    .hero{text-align:center;max-width:980px;margin:28px auto 10px auto}
    .hero h1{font-size:38px;font-weight:800;color:var(--brand)}
    .hero p{color:var(--muted)}
    .grid{max-width:1280px;margin:18px auto;display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
    .grid-3{max-width:1280px;margin:6px auto 28px;display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
    .card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px}
    .card h3{margin:0 0 10px 0;color:var(--ink)}
    .btn{margin-top:12px;background:var(--brand);color:#fff;border:none;border-radius:12px;padding:12px 16px;font-weight:800;cursor:pointer;box-shadow:var(--shadow)}
    .btn-secondary{margin-top:8px;background:#fff;color:var(--brand);border:1px solid var(--brand);border-radius:10px;padding:8px 12px;font-weight:700;cursor:pointer}
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Vimal AI Outfit App") as demo:
        # Header
        gr.HTML("""
        <div class="header">
          <div class="brand"><span class="logo"></span> Vimal AI Outfit App</div>
          <nav class="nav"><a>Try On</a><a>Shop</a><a>Outfits</a><a>About</a></nav>
          <div>🔍 🛍️ 👤</div>
        </div>
        """)

        # Hero
        gr.HTML("""
        <section class="hero">
          <h1>AI Virtual Try-On Studio</h1>
          <p>Upload, capture, or paste an image URL for person and garments. Supports applying both Upper and Lower in one go.</p>
        </section>
        """)

        # -------- TOP ROW: Person / Generated / Advanced Options --------
        with gr.Row(elem_classes="grid"):
            # Person
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Person Image")
                person_image = gr.Image(
                    sources=["upload", "webcam"],
                    type="filepath",
                    label="",
                    height=260
                )
                person_url = gr.Textbox(label="Or paste Person Image URL", placeholder="https://example.com/photo.jpg")
                load_person_btn = gr.Button("Load Person from URL", elem_classes="btn-secondary")
                gr.Examples(inputs=person_image, examples=person_images, examples_per_page=6)

            # Generated
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Generated Image")
                output_image = gr.Image(label="", height=260)
                generate_btn = gr.Button("Generate", elem_classes="btn")
                with gr.Accordion("Debug", open=False):
                    vt_mask = gr.Image(label="Mask", height=150)
                    vt_densepose = gr.Image(label="DensePose", height=150)

            # Advanced Options
            with gr.Column(elem_classes="card"):
                gr.Markdown("### ⚙️ Advanced Options")
                vt_model_type = gr.Radio(
                    label="Model Type",
                    choices=[("VITON-HD (Recommended)", "viton_hd"),
                             ("DressCode (Experimental)", "dress_code")],
                    value="dress_code",   # default DressCode
                )
                vt_garment_type = gr.Radio(
                    label="Garment Mode",
                    choices=[("Upper", "upper_body"),
                             ("Lower", "lower_body"),
                             ("Dress", "dresses"),
                             ("Both (Upper + Lower)", "both")],
                    value="upper_body",
                )
                vt_ref_acceleration = gr.Checkbox(label="Accelerate Reference UNet", value=False)
                vt_repaint = gr.Checkbox(label="Repaint Mode", value=False)
                vt_step = gr.Number(label="Inference Steps", value=30, minimum=30, maximum=100, step=1)
                vt_scale = gr.Number(label="Guidance Scale", value=2.5, minimum=0.1, maximum=5.0, step=0.1)
                vt_seed = gr.Number(label="Random Seed", value=42, minimum=-1, maximum=2147483647, step=1)

        # -------- SECOND ROW: Separate Garment Blocks (Upper / Lower / Dress) --------
        with gr.Row(elem_classes="grid-3"):
            with gr.Column(elem_classes="card"):
                gr.Markdown("### Upper Garment")
                garment_upper = gr.Image(
                    sources=["upload", "webcam"],
                    type="filepath",
                    label="Upload/Capture Upper Garment",
                    height=240
                )
                upper_url = gr.Textbox(label="Or paste Upper Garment URL", placeholder="https://example.com/top.png")
                load_upper_btn = gr.Button("Load Upper from URL", elem_classes="btn-secondary")
                gr.Examples(inputs=garment_upper, examples=garment_examples, examples_per_page=6)

            with gr.Column(elem_classes="card"):
                gr.Markdown("### Lower Garment")
                garment_lower = gr.Image(
                    sources=["upload", "webcam"],
                    type="filepath",
                    label="Upload/Capture Lower Garment",
                    height=240
                )
                lower_url = gr.Textbox(label="Or paste Lower Garment URL", placeholder="https://example.com/bottom.jpg")
                load_lower_btn = gr.Button("Load Lower from URL", elem_classes="btn-secondary")
                gr.Examples(inputs=garment_lower, examples=garment_examples, examples_per_page=6)

            with gr.Column(elem_classes="card"):
                gr.Markdown("### Dress")
                garment_dress = gr.Image(
                    sources=["upload", "webcam"],
                    type="filepath",
                    label="Upload/Capture Dress",
                    height=240
                )
                dress_url = gr.Textbox(label="Or paste Dress URL", placeholder="https://example.com/dress.webp")
                load_dress_btn = gr.Button("Load Dress from URL", elem_classes="btn-secondary")
                gr.Examples(inputs=garment_dress, examples=garment_examples, examples_per_page=6)

        # ---------- URL Loaders ----------
        def load_url_to_image(url: str):
            return _download_image_to_tmp(url)

        load_person_btn.click(fn=load_url_to_image, inputs=[person_url], outputs=[person_image])
        load_upper_btn.click(fn=load_url_to_image, inputs=[upper_url], outputs=[garment_upper])
        load_lower_btn.click(fn=load_url_to_image, inputs=[lower_url], outputs=[garment_lower])
        load_dress_btn.click(fn=load_url_to_image, inputs=[dress_url], outputs=[garment_dress])

        # ---------- Generate Handler ----------
        def run_vton(
            src_path, gmode, up_path, low_path, drs_path,
            vt_ref_acceleration, vt_step, vt_scale, vt_seed, vt_model_type, _vt_garment_type, vt_repaint
        ):
            """
            gmode:
              - 'upper_body'  -> single pass with upper
              - 'lower_body'  -> single pass with lower
              - 'dresses'     -> single pass with dress
              - 'both'        -> TWO passes: upper first, then lower
            """
            _ensure(src_path, "Please provide a Person Image (upload/capture or URL).")

            if gmode == "both":
                _ensure(up_path, "Please provide the Upper garment (upload/capture or URL) for 'Both' mode.")
                _ensure(low_path, "Please provide the Lower garment (upload/capture or URL) for 'Both' mode.")

                # Pass 1: apply UPPER
                gen1, _, _ = leffa_predictor.leffa_predict_vt(
                    src_image_path=src_path,
                    ref_image_path=up_path,
                    ref_acceleration=bool(vt_ref_acceleration),
                    step=int(vt_step),
                    scale=float(vt_scale),
                    seed=int(vt_seed),
                    vt_model_type=vt_model_type,
                    vt_garment_type="upper_body",
                    vt_repaint=bool(vt_repaint),
                )
                inter_path = _save_np_image_to_tmp(gen1)

                # Pass 2: apply LOWER on result of Pass 1
                gen2, mask2, dp2 = leffa_predictor.leffa_predict_vt(
                    src_image_path=inter_path,
                    ref_image_path=low_path,
                    ref_acceleration=bool(vt_ref_acceleration),
                    step=int(vt_step),
                    scale=float(vt_scale),
                    seed=int(vt_seed),
                    vt_model_type=vt_model_type,
                    vt_garment_type="lower_body",
                    vt_repaint=bool(vt_repaint),
                )
                return gen2, mask2, dp2

            elif gmode == "upper_body":
                _ensure(up_path, "Please provide the Upper garment (upload/capture or URL).")
                return leffa_predictor.leffa_predict_vt(
                    src_image_path=src_path,
                    ref_image_path=up_path,
                    ref_acceleration=bool(vt_ref_acceleration),
                    step=int(vt_step),
                    scale=float(vt_scale),
                    seed=int(vt_seed),
                    vt_model_type=vt_model_type,
                    vt_garment_type="upper_body",
                    vt_repaint=bool(vt_repaint),
                )

            elif gmode == "lower_body":
                _ensure(low_path, "Please provide the Lower garment (upload/capture or URL).")
                return leffa_predictor.leffa_predict_vt(
                    src_image_path=src_path,
                    ref_image_path=low_path,
                    ref_acceleration=bool(vt_ref_acceleration),
                    step=int(vt_step),
                    scale=float(vt_scale),
                    seed=int(vt_seed),
                    vt_model_type=vt_model_type,
                    vt_garment_type="lower_body",
                    vt_repaint=bool(vt_repaint),
                )

            elif gmode == "dresses":
                _ensure(drs_path, "Please provide the Dress (upload/capture or URL).")
                return leffa_predictor.leffa_predict_vt(
                    src_image_path=src_path,
                    ref_image_path=drs_path,
                    ref_acceleration=bool(vt_ref_acceleration),
                    step=int(vt_step),
                    scale=float(vt_scale),
                    seed=int(vt_seed),
                    vt_model_type=vt_model_type,
                    vt_garment_type="dresses",
                    vt_repaint=bool(vt_repaint),
                )

            else:
                raise gr.Error("Unknown garment mode. Please select Upper, Lower, Dress, or Both.")

        generate_btn.click(
            fn=run_vton,
            inputs=[
                person_image, vt_garment_type, garment_upper, garment_lower, garment_dress,
                vt_ref_acceleration, vt_step, vt_scale, vt_seed, vt_model_type, vt_garment_type, vt_repaint
            ],
            outputs=[output_image, vt_mask, vt_densepose]
        )

        gr.Markdown(
            "<div style='text-align:center;color:#6b7280;margin:18px 0;'>"
            "Tip: For Google Drive/Share links, ensure a direct-download URL. "
            "Models are trained on academic datasets (VITON-HD / DressCode) for research/demo purposes."
            "</div>"
        )

        demo.launch(share=True, server_port=7860, allowed_paths=["./ckpts/examples"])
