from flask import Flask, request, jsonify
#from flask_cors import CORS, cross_origin
from model import load_model, inference
import PIL.Image as Image
import io
import os
import numpy as np
app = Flask(__name__)
#cors = CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'
model = load_model()
@app.route("/process", methods = ["POST"])
#@cross_origin()
def process():
    file = request.files['file']
    image = np.array(Image.open(io.BytesIO(file.read())).convert("RGB"))
    pred_ing, pred_class= inference(model, image)
    output = {"Ingredient": pred_ing, "Class": pred_class}
    return jsonify(output)

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = False, port = "5000")