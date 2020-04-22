from flask import Flask, request
from PIL import Image
import io
import numpy as np
import cv2
from potrait.getPotrait import Potrait
app = Flask(__name__)


@app.route("/hello", methods=["POST"])
def func():
    # return "satinder"
    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            img = np.array(image)
            # img=cv2.resize(img,(500,500))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            obj = Potrait()
            obj.getPotrait(img)
            # cv2.imshow("ds",img)
            # cv2.waitKey(1000)
            # image.show()
            return "success"


if __name__ == "__main__":
    app.run(debug=True)
