from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from helpers import *

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def home():
    # delete_uploaded_files()
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            wav_file = convert_to_wav(filepath)
            spectrogram_file, prediction = convert_to_spectrogram(wav_file)
            print(prediction)
            features = extract_and_process_features(wav_file)
            print(features)
            result = make_prediction(features)
            if prediction <= 0:
                label = "Real"
            else:
                label = "Fake!"
                
            return render_template('result.html', spectrogram=spectrogram_file, label=label,filename=filename,result=result)
    return redirect(url_for('home'))

@app.route('/contact')
def contact():
    return render_template('contact_us.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
