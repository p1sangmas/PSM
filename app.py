from flask import Flask, request, send_file
import subprocess
import os
import requests

app = Flask(__name__)

# Google Drive file URL (replace with your file's shareable link)
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID"
MODEL_PATH = os.path.join("modelscope", "damo", "cv_ddcolor_image-colorization", "pytorch_model.pt")

def download_file(url, save_path):
    """
    Download a file from a URL and save it to the specified path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad status codes

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the file
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@app.route("/colorize", methods=["POST"])
def colorize():
    try:
        # Save uploaded file
        uploaded_file = request.files["file"]
        input_path = os.path.join("assets", "test_images", "uploaded_image.JPEG")
        os.makedirs(os.path.dirname(input_path), exist_ok=True)  # Create directory if it doesn't exist
        uploaded_file.save(input_path)

        # Download the model file if it doesn't exist
        if not os.path.exists(MODEL_PATH):
            print("Downloading model file...")
            download_file(MODEL_URL, MODEL_PATH)
            print("Model file downloaded.")

        # Create the output directory if it doesn't exist
        output_path = os.path.join("colorize_output", "colorized_image.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run the colorization script
        command = [
            "python3", os.path.join("inference", "colorization_pipeline.py"),
            "--input_file", input_path,
            "--output_file", output_path,
            "--model_path", MODEL_PATH
        ]
        subprocess.run(" ".join(command), shell=True)

        # Return the colorized image
        return send_file(output_path, mimetype="image/png")

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing the image.", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
