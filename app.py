from flask import Flask, request, render_template, send_file
import os
import shutil
from werkzeug.utils import secure_filename
from inpainting import inpaint_image
from PIL import Image
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    # Clean up old files in outputs
    for file in os.listdir(app.config['OUTPUT_FOLDER']):
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    return render_template('index.html')

@app.route('/inpaint', methods=['POST'])
def inpaint():
    # File upload handling
    if 'image' not in request.files or 'mask' not in request.files:
        return "Please upload both an image and a mask", 400
    image_file = request.files['image']
    mask_file = request.files['mask']
    
    if image_file.filename == '' or mask_file.filename == '':
        return "No selected file", 400
    
    if not (allowed_file(image_file.filename) and allowed_file(mask_file.filename)):
        return "Invalid file type. Only PNG, JPG, JPEG allowed", 400

    # Save uploaded files temporarily
    image_filename = secure_filename(str(uuid.uuid4()) + '_' + image_file.filename)
    mask_filename = secure_filename(str(uuid.uuid4()) + '_' + mask_file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
    image_file.save(image_path)
    mask_file.save(mask_path)

    # Get parameters
    num_iter = int(request.form.get('num_iter', 3000))
    lr = float(request.form.get('lr', 0.01))

    # Process image
    try:
        out_pil = inpaint_image(image_path, mask_path, num_iter=num_iter, lr=lr)
        output_filename = str(uuid.uuid4()) + '_inpainted.png'
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        out_pil.save(output_path)

        # Copy original and mask to outputs with error handling
        base_uuid = str(uuid.uuid4())
        output_image_filename = f"{base_uuid}_original.{image_file.filename.rsplit('.', 1)[1].lower()}"
        output_mask_filename = f"{base_uuid}_mask.{mask_file.filename.rsplit('.', 1)[1].lower()}"
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_image_filename)
        output_mask_path = os.path.join(app.config['OUTPUT_FOLDER'], output_mask_filename)
        shutil.copy2(image_path, output_image_path)
        shutil.copy2(mask_path, output_mask_path)

        # Clean up uploaded files
        os.remove(image_path)
        os.remove(mask_path)
        
        print(f"Rendering: input_image={output_image_filename}, mask_image={output_mask_filename}, output_image={output_filename}")
        return render_template('result.html', 
                             input_image=output_image_filename,
                             mask_image=output_mask_filename,
                             output_image=output_filename)
    except Exception as e:
        return f"Error processing image: {str(e)}", 500

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)