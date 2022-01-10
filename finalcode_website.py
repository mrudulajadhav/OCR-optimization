import cv2
import numpy as np
import argparse
from imutils.object_detection import non_max_suppression
import pytesseract as tess
from itertools import permutations 
import re
import pandas as pd
from openpyxl import load_workbook
tess.pytesseract.tesseract_cmd = "tesseract.exe"


# Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < 0.5:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)


def sort_contours(boundingBoxes, method="left-to-right"):
	# initialize the reverse flag and sort index
    reverse = False
    i = 0
	# handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
        
	# return the list of sorted bounding boxes
    boundingBoxes = sorted(boundingBoxes,key=lambda b:b[i], reverse=reverse)
    return (boundingBoxes)





def ocr_(input_img):
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    half_h = img.shape[1]//2 #dimension cropping image 

    crop = img[:367, half_h:] #cropping upper right of the bill
    crop2 = img[:367, half_h-15:] #for cornercases

    ret,bininv = cv2.threshold(crop,200,255,cv2.THRESH_BINARY_INV)#binary inversion
    cv2.imwrite("temp.png",bininv)

    image = cv2.imread("temp.png")

    #Saving a original image and shape
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new height and width to default 320 by using args #dictionary.  
    (newW, newH) = (320, 320)

    #Calculate the ratio between original and new image for both height and weight. 
    #This ratio will be used to translate bounding box location on the original image. 
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the original image to new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	    (123.68, 116.78, 103.94), swapRB=True, crop=False)
    # load the pre-trained EAST model for text detection 
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # The following two layer need to pulled from EAST model for achieving this. 
    layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
  
    #Forward pass the blob from the image to get the desired output layers
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)


    # Find predictions and  apply non-maxima suppression
    (boxes, confidence_val) = predictions(scores, geometry)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
    # loop over the bounding boxes to find the coordinate of bounding boxes
    bb = []
    for (startX, startY, endX, endY) in boxes:
        # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        bb.append((startX, startY, endX, endY))
    cv2.imwrite("EAST.png",orig)

    #bb is the list of bounding boxes detected by EAST 
    boundingBoxes = bb

    #EAST.png is image after text detection by EAST detector
    img = cv2.imread("EAST.png")

    args = {"image":"EAST.png", "method":"top-to-bottom"}

    # sort the boundingBoxes according to the provided method
    (boundingBoxes) = sort_contours(boundingBoxes, method=args["method"])

    text_list = []

    #put sorted numbers on the image
    for (i,c) in enumerate(boundingBoxes):
        cv2.putText(img, "{}".format(i), (c[0],c[1]), cv2.FONT_HERSHEY_SIMPLEX,0.4, (0, 0, 255), 2)
        if(c[0]<0):
            continue
        r = crop[c[1]:c[3], c[0]:c[2]]
        configuration = ("-l eng --oem 1 --psm 8")
        text = tess.image_to_string(r,config = configuration)
        text_list.append(text)

    strr = " ".join(text_list)
    text = strr.split("\n\x0c")


    #maintaining a list of punctuations to remove them
    marks = ":! ‘$“-(|}—,"
    for i in range(len(text)):
        for x in text[i]:  
            if x in marks:  
                text[i] = text[i].replace(x, "")  
    flag = 0 #0 for dd/mm/yyyy format and 1 for mmmdd yyyy format
    #print(text)

    store_date = 'Not Present'
    store_amt = 'Not Present'
    for i  in range(len(text)-3):
        slic = (text[i:i+4])

        # Get all permutations 
        perm = permutations([slic[0], slic[1], slic[2], slic[3]]) 
  
     # Print the obtained permutations, returns tuple, convert to list
        for i in list(perm): 
            t = list(i)
            if(t[0].upper()==('DUE') and t[1].upper()==('DATE')):
                x = re.search(r"^([0-2][0-9]|(3)[0-1])(\/)(((0)[0-9])|((1)[0-2]))(\/)\d{4}", t[2])
                if(x == None):
                    x = re.search(r"^(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}", t[2])
                    if(x == None):
                        break
                    else:
                        if(re.search(r"^(\d{4})",t[3]) != None):
                            flag = 1
                if(flag == 0):
                    print(" Due Date: {}".format(x.group()))
                    store_date = x.group()
                else:
                    print(" Due Date: {} {}".format(x.group(),t[3]))
                    store_date = x.group()+' '+t[3]
                break
    #if due date value is below its key
    _k = 0
    _lis = []
    _short_list = []
    date_key = False

    for (i,c) in enumerate(boundingBoxes):
        if(c[0] < 0):
            continue
        _lis.extend([[boundingBoxes[i],text[_k]]])
        _k += 1

    for _index in range(len(_lis)):
        if('DUEDAT' in _lis[_index][1].upper()):
            _num = _index
            date_key = True
            break
    if(date_key == True):
        for i in range(_num,len(_lis)):
            _short_list.extend([[_lis[i][0],_lis[i][1]]])

        _final = []
        for i in range(len(_short_list)):
            if(_short_list[i][0][0] <= _short_list[0][0][0]):
                _final.append(_short_list[i])

        for tem in _final:
            if "/" in tem[1]:
                coord = tem[0]
                cut = crop2[tem[0][1]:tem[0][3], tem[0][0]+5:tem[0][2]+20]
                configuration = ("-l eng --oem 1 --psm 8")
                date = tess.image_to_string(cut,config = configuration)
                for x in date:  
                    if x in marks:  
                        date = date.replace(x, "") 
                print(" Due Date: {}".format(date.strip()))
                store_date = date.strip()

    k = 0
    lis = []
    short_list = []
    keyword = False #for amount keyword

    for (i,c) in enumerate(boundingBoxes):
        if(c[0] < 0):
            continue
        lis.extend([[boundingBoxes[i],text[k]]])
        k += 1

    for index in range(len(lis)):
        if('AMOU' in lis[index][1].upper() or 'ROUND' in lis[index][1].upper()):
            keyword = True #keyword is present
            num = index
            break

    if(keyword == True):#if keyword is present, then proceed
        for i in range(num,len(lis)):
            short_list.extend([[lis[i][0],lis[i][1]]])
        final = []
        for i in range(len(short_list)):
            if(short_list[i][0][0] >= short_list[0][0][0]-10):
                final.append(short_list[i])
    else: #if keyword is absent, we dont detect amount
        pass

    #vertical logic for amount extraction
    vertical = False  
    if(keyword == True):
        for item in final:
            if ("." in item[1]):
                res = item[1].replace('.', '').isdigit()
            else:
                res = item[1].isdigit()
            if(res == True): 
                new_r = crop[item[0][1]:item[0][3], item[0][0]+5:item[0][2]+6]
                configuration = ("-l eng --oem 1 --psm 8")
                val = tess.image_to_string(new_r,config = configuration)
                for x in val:  
                    if x in marks:  
                        val = val.replace(x, "") 
                u = val.strip() #to remove unknown character
                print(" Amount Payable: {}".format(u.strip(".")))#to remove . at the end
                store_amt = u.strip(".")
                vertical = True
                break
            else:
                pass

    #if vertical didn't work, try horizontal logic
    if(vertical == False and keyword == True):
        for i  in range(len(text)-1):
            slicc = (text[i:i+2])

        # Get all permutations 
            amt_perm = permutations([slicc[0], slicc[1]]) 
  
     # Print the obtained permutations, returns tuple, convert to list
            for i in list(amt_perm): 
                t = list(i)
                if("AMOUNT" in t[0].upper()):
                    if ("." in t[1]):
                        res = t[1].replace('.', '').isdigit()
                    else:
                        res = t[1].isdigit()
                    if(res == True):
                        print(" Amount Payable: {}".format(t[1].strip(".")))#removes dot at the end if any
                        store_amt = t[1].strip(".")
                    else:
                        break

    # new dataframe with same columns
    df = pd.DataFrame({'Due Date': [store_date],
                       'Amount Payable': [store_amt]})
    writer = pd.ExcelWriter('demo.xlsx', engine='openpyxl')
    # try to open an existing workbook
    writer.book = load_workbook('demo.xlsx')
    # copy existing sheets
    writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
    # read existing file
    reader = pd.read_excel(r'demo.xlsx')
    # write out the new sheet
    df.to_excel(writer,index=False,header=False,startrow=len(reader)+1)

    writer.close()

    return store_date, store_amt