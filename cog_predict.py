# flake8: noqa
# This file is used for deploying replicate models
# running: cog predict -i img=@inputs/00017_gray.png -i version='General - v3' -i scale=2 -i face_enhance=True -i tile=0
# push: cog push r8.im/xinntao/realesrgan
import os
import cv2
import shutil
import tempfile
import torch
from io import BytesIO
import base64
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
import numpy as np

os.system('python setup.py develop')


try:
    from realesrgan.utils import RealESRGANer
    from cog import BasePredictor, Input, Path
except Exception:
    print('please install cog and realesrgan package')


class Predictor(BasePredictor):

    def setup(self):
        os.makedirs('output', exist_ok=True)
        if not os.path.exists('weights/RealESRGAN_x4plus.pth'):
            os.system(
                'wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./weights'
            )
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = 'weights/RealESRGAN_x4plus.pth'
        self.upsampler = RealESRGANer(
            scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=True)

    def predict(
        self,
        img: str = Input(description='Input Image'),
        scale: float = Input(description='Rescaling factor', default=2),
    ) -> dict:
        try:
            init_image = Image.open(BytesIO(base64.b64decode(img))).convert("RGB")
            extension = "png"
            img = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            elif len(img.shape) == 2:
                img_mode = None
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_mode = None

            h, w = img.shape[0:2]
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            try:
                output, _ = self.upsampler.enhance(img, outscale=scale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set "tile" to a smaller size, e.g., 400.')

            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            out_path = Path(tempfile.mkdtemp()) / f'out.{extension}'
            cv2.imwrite(str(out_path), output)
            return dict(data=out_path)
        except Exception as error:
            print('global exception: ', error)
        finally:
            clean_folder('output')


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
