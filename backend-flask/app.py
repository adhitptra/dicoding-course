from flask import Flask, request, jsonify
import tensorflow as tf
from werkzeug.utils import secure_filename
from google.cloud import firestore

app = Flask(__name__)

# Load model from Cloud Storage (replace with your bucket URL)
model = tf.keras.models.load_model('https://storage.googleapis.com/submissionmlgc-adhitya/model.json')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"status": "fail", "message": "No image part"}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({"status": "fail", "message": "No selected file"}), 400
    if image.content_length > 1000000:
        return jsonify({
            "status": "fail",
            "message": "Payload content length greater than maximum allowed: 1000000"
        }), 413
    
    filename = secure_filename(image.filename)
    image.save(filename)

    # Preprocess the image
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    result = "Cancer" if prediction > 0.5 else "Non-cancer"
    suggestion = "Segera periksa ke dokter!" if result == "Cancer" else "Penyakit kanker tidak terdeteksi."

    # Save result to Firestore
    db = firestore.Client()
    doc_ref = db.collection('predictions').document()
    doc_ref.set({
        'id': doc_ref.id,
        'result': result,
        'suggestion': suggestion,
        'createdAt': firestore.SERVER_TIMESTAMP
    })

    return jsonify({
        "status": "success",
        "message": "Model is predicted successfully",
        "data": {
            "id": doc_ref.id,
            "result": result,
            "suggestion": suggestion,
            "createdAt": str(doc_ref.get().to_dict()['createdAt'])
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)