from flask import Flask, request, jsonify, send_file
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from PIL import Image
from werkzeug.exceptions import NotFound
import torch
import numpy as np
import cv2
import os

app = Flask(__name__)

controlnet_conditioning_scale = 0.8

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0-small",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        data = request.get_json()
        prompt = data['prompt']
        negative_prompt = data['negative_prompt']
        image_path = data['image_path']

        image = load_image(image_path)
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        images = pipe(
            prompt, negative_prompt=negative_prompt, image=image, controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        output_path = "output.png"
        images[0].save(output_path)

        return jsonify({'message': 'Image generated successfully!', 'output_path': output_path})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#Ruta get
@app.route('/get_image', methods=['GET'])
def get_image():
    try:
        # Obtener la ruta de la imagen generada desde la solicitud GET
        output_path = request.args.get('output_path')

        # Verificar si la ruta de la imagen existe
        if os.path.exists(output_path):
            # Devolver la imagen como respuesta
            return send_file(output_path, mimetype='image/png')
        else:
            raise NotFound("Image not found")

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Ruta de inicio, por ejemplo, para comprobar si la API está en línea
@app.route('/')
def index():
    return 'API is up and running'

if __name__ == '__main__':
    app.run(debug=True)










