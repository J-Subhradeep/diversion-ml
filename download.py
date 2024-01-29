import timm
from fastai.vision.all import *
from io import BytesIO
from PIL import Image
import pickle
from huggingface_hub import from_pretrained_fastai, _save_pretrained_fastai
from contextlib import contextmanager
import pathlib


@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

with set_posix_windows():
    potato_cat = ('Early_Blight', 'Healthy', 'Late_Blight')
    potato = from_pretrained_fastai("Luna-Skywalker/potato_dtect")

    def classify_image(categories,model,img):
        pred_class, pred_idx, probs = model.predict(img)
        return dict(zip(categories,map(float,probs)))
    
    img = 'LateBlight03.jpg'
    prediction = classify_image(potato_cat,potato,img)
    print(prediction)
    _save_pretrained_fastai(potato, "ml_models\potato_dtect")

    corn = from_pretrained_fastai("Luna-Skywalker/corn_dtect")
    wheat = from_pretrained_fastai("Luna-Skywalker/wheat_dtect")
    rice = from_pretrained_fastai("Luna-Skywalker/rice_dtect")
    _save_pretrained_fastai(corn, "ml_models\corn_dtect")
    _save_pretrained_fastai(wheat, "ml_models\wheat_dtect")
    _save_pretrained_fastai(rice, "ml_models\Rice_dtect")