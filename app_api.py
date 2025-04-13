import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import gradio as gr

# Download checkpoints
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")

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
        pt_model = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-xl-1.0-inpainting-0.1",
            pretrained_model="./ckpts/pose_transfer.pth",
            dtype="float16",
        )
        self.pt_inference = LeffaInference(model=pt_model)

    def leffa_predict_vt(self, src_image_path, ref_image_path, preprocess_garment=False, vt_model_type="viton_hd", vt_garment_type="upper_body", step=30, scale=2.5, seed=42):
        # Virtual Try-On Prediction Logic
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "virtual_tryon",
            False,
            step,
            scale,
            seed,
            vt_model_type,
            vt_garment_type,
            False,
            preprocess_garment,
        )

    def leffa_predict_pt(self, src_image_path, ref_image_path, step=30, scale=2.5, seed=42):
        # Pose Transfer Prediction Logic
        return self.leffa_predict(
            src_image_path,
            ref_image_path,
            "pose_transfer",
            False,
            step,
            scale,
            seed,
        )

    def leffa_predict(self, src_image_path, ref_image_path, control_type, ref_acceleration, step, scale, seed, vt_model_type=None, vt_garment_type=None, vt_repaint=None, preprocess_garment=None):
        # Core Prediction Logic (same as in the original script)
        pass  # Replace with the actual implementation from the original script


# Initialize Predictor
leffa_predictor = LeffaPredictor()

# Define API Endpoints
with gr.Blocks() as api_app:
    with gr.Tab("API Endpoints"):
        # Virtual Try-On API
        with gr.Row():
            vt_src_image = gr.File(label="Person Image")
            vt_ref_image = gr.File(label="Garment Image")
            vt_preprocess_garment = gr.Checkbox(label="Preprocess Garment", value=False)
            vt_model_type = gr.Radio(choices=["viton_hd", "dress_code"], label="Model Type", value="viton_hd")
            vt_garment_type = gr.Radio(choices=["upper_body", "lower_body", "dresses"], label="Garment Type", value="upper_body")
            vt_step = gr.Number(label="Inference Steps", value=30)
            vt_scale = gr.Number(label="Guidance Scale", value=2.5)
            vt_seed = gr.Number(label="Random Seed", value=42)
            vt_gen_button = gr.Button("Generate Virtual Try-On")
            vt_output = gr.Image(label="Generated Image")

        vt_gen_button.click(
            fn=leffa_predictor.leffa_predict_vt,
            inputs=[vt_src_image, vt_ref_image, vt_preprocess_garment, vt_model_type, vt_garment_type, vt_step, vt_scale, vt_seed],
            outputs=vt_output
        )

        # Pose Transfer API
        with gr.Row():
            pt_src_image = gr.File(label="Source Person Image")
            pt_ref_image = gr.File(label="Target Pose Person Image")
            pt_step = gr.Number(label="Inference Steps", value=30)
            pt_scale = gr.Number(label="Guidance Scale", value=2.5)
            pt_seed = gr.Number(label="Random Seed", value=42)
            pt_gen_button = gr.Button("Generate Pose Transfer")
            pt_output = gr.Image(label="Generated Image")

        pt_gen_button.click(
            fn=leffa_predictor.leffa_predict_pt,
            inputs=[pt_src_image, pt_ref_image, pt_step, pt_scale, pt_seed],
            outputs=pt_output
        )

# Launch API
api_app.launch(server_port=7860, api_open=True)