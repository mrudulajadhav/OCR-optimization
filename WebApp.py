from flask import *
import cv2
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
from ocr import *

import os

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = "super secret key"

path = os.path.abspath("Images")

@app.route('/', methods = ['POST','GET'])
def home():
    return render_template('index1.html')

@app.route('/success', methods = ['POST'])  
def success(): 
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print("NO FILE")
            flash('No file part')
            amt = 0
            due = 0
            return redirect(url_for('home'))
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('home'))
        else:
            f = request.files['file']
            f.save(os.path.join(path,f.filename))   
            print(f.filename)
        
            img = cv2.imread(os.path.join(path,f.filename))
            amt, due = ocr_(img)
            return render_template('success.html', due_date = due, amount = amt) 

    
if __name__ == "__main__":
    app.run()