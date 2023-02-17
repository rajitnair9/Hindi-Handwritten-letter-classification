from flask import Flask, request, render_template
import tensorflow as tf
import os
import numpy as np
import cv2

app = Flask(__name__,static_folder='static')
app.config['UPLOAD_FOLDER'] = 'E:\\Flask\\myenv\\Classification_Hindi\\static\\uploads'
# Load the model
model = tf.keras.models.load_model('E:\\Flask\\myenv\\Classification_Hindi\\model.h5')

# Define the label map
label_map = {0: 'च', 1: 'छ', 2: 'ग', 3: 'ध', 4: 'ज', 5: 'झ', 6: 'क', 7: 'ख', 8: 'ङ', 9: 'ञ'}

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image = cv2.imread(filepath)
        image = cv2.resize(image, (32, 32))
        #image = image.astype("float") / 255.0
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        predicted_label = np.argmax(prediction)
        predicted_character = label_map[predicted_label]
        return render_template('result.html', letter=predicted_character, image_file=file.filename)
    return render_template('upload.html')
if __name__ == '__main__':
    app.run(debug=True)
