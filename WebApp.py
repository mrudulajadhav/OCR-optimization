from flask import *
import cv2
import numpy as np
import argparse
from imutils.object_detection import non_max_suppression
import pytesseract as tess
from itertools import permutations 
import re
import pandas as pd
from openpyxl import load_workbook
tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from finalcode_website import *

import os

app = Flask(__name__)
app.static_folder = 'static'

path = os.path.abspath("")

@app.route('/')
def hello():
    return render_template('index1.html')

@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)   
        print(f.filename)
        
        img = cv2.imread(path+"\\"+f.filename)
        amt, due = ocr_(img)
        return render_template('success.html', due_date = due, amount = amt)  
    
if __name__ == "__main__":
    app.run()