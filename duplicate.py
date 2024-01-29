from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import uvicorn
import timm
from fastai.vision.all import *
from io import BytesIO
from PIL import Image
import pickle
from contextlib import contextmanager
import pathlib

# For error "NotImplementedError: cannot instantiate 'PosixPath' on your system"
@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


with set_posix_windows():
    model_paths = ["./ml_models/corn_dtect/model.pkl", "./ml_models/wheat_dtect/model.pkl", "./ml_models/potato_dtect/model.pkl","./ml_models/Rice_dtect/model.pkl"]
    models = [load_learner(model_path) for model_path in model_paths]

    corn_cat = ('Common_Rust', 'Gray_Leaf_Spot', 'Healthy', 'Northern_Leaf_Blight')
    potato_cat = ('Early_Blight', 'Healthy', 'Late_Blight')
    rice_cat =('Brown_Spot', 'Neck_Blast', 'Leaf_Blast', 'Healthy')
    wheat_cat = ('Healthy', 'Yellow_Rust', 'Brown_Rust')
    cat = [corn_cat,wheat_cat,potato_cat,rice_cat]

    # prediction function
    def classify_image(categories,model,img):
        pred_class, pred_idx, probs = model.predict(img)
        return dict(zip(categories,map(float,probs)))
    
    def predict_image(file, cat, model):
            # Read the image file

            try:
                image = PILImage.create(file)
                if image is None:
                    raise ValueError("Image could not be loaded.")
            except Exception as e:
                    print(f"Error opening image: {e}")
            print(image)

            # Perform prediction using the model
            prediction = classify_image(cat,model,image)

            # Prepare the response
            # response = {
            #     "model_name": model.name,
            #     "prediction": str(prediction)
            # }

            # Return JSON response
            return JSONResponse(content=jsonable_encoder(prediction), status_code=200)


    # Create FastAPI app
    app = FastAPI()
    #middleware config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/corn")
    async def corn(file: UploadFile = File(...)):
        contents = await file.read()
        return predict_image(contents , corn_cat, models[0])

    @app.post("/wheat")
    async def wheat(file: UploadFile = File(...)):
        contents = await file.read()
        return predict_image(contents, wheat_cat, models[1])

    @app.post("/potato")
    async def potato(file: UploadFile = File(...)):
        contents = await file.read()
        return predict_image(contents, potato_cat, models[2])

    @app.post("/rice")
    async def rice(file: UploadFile = File(...)):
        contents = await file.read()
        return predict_image(contents, rice_cat, models[3])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)