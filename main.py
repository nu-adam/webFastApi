from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the file
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(save_path)

    return jsonify({"message": f"File uploaded successfully", "path": save_path})

if __name__ == '__main__':
    app.run(debug=True)
