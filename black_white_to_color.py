
from flask import Flask, render_template, request
import cv2
import numpy as np
import base64

app = Flask(__name__)

def blend_images(original, colorized, alpha):
    return cv2.addWeighted(original, alpha, colorized, 1 - alpha, 0)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/', methods=['POST'])

def colorize_image():
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

    net = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
    pts = np.load('pts_in_hull.npy')

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)

    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    _, original_encoded = cv2.imencode('.png', image)
    original_base64 = base64.b64encode(original_encoded).decode('utf-8')

    _, colorized_encoded = cv2.imencode('.png', colorized)
    colorized_base64 = base64.b64encode(colorized_encoded).decode('utf-8')

    return render_template('index.html', original_base64=original_base64, colorized_base64=colorized_base64)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000,debug=True)
