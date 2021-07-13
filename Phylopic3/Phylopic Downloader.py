import json
import requests
import cv2
import numpy as np
from PIL import Image


total = json.loads(requests.get('http://phylopic.org/api/a/image/count').text)["result"]


images = json.loads(requests.get('http://phylopic.org/api/a/image/list/0/'+str(total)).text)["result"]




index = 39
while (index < total):
    uid= images[index]["uid"]
    directNames= json.loads(requests.get("http://phylopic.org/api/a/image/" + str(uid), params={"options":"directNames"}).text)["result"]["directNames"]
    if (len(directNames)==0):
        index+=1
        continue
    nameID= directNames[0]["uid"]
    lineage = requests.get("http://phylopic.org/api/a/name/" + str(nameID) + "/taxonomy", params = {"supertaxa": "all"}).text
    # if ("68424967-5109-4f0d-a8e2-77e7edbe94ab" in lineage): #Kingdom Metazoa far too broad
    if ("68226175-f88d-4ea8-8228-3204c49bfda0" in lineage):
        print("Its an nephrazoan!")
    else:
        print("its not an nephrazoan :(")
        index+=1
        continue



    img_data = requests.get("http://phylopic.org/assets/images/submissions/" + str(uid) + ".256.png").content

    # CV2
    nparr = np.frombuffer(img_data, np.uint8)
    alpha_img = cv2.imdecode(nparr, cv2.cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_COLOR in OpenCV 3.1


    try:
        alpha_channel = alpha_img[:, :, 3]
    except Exception as badBoi:
        print("Error loading " + str(index))
        index+=1
        continue
    _, mask = cv2.threshold(alpha_channel, 40, 255, cv2.THRESH_BINARY)  # binarize mask
    color = alpha_img[:, :, :3]
    img = cv2.bitwise_not(cv2.bitwise_not(color, mask=mask))

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # If you want color

    # read image
    # img = cv2.imread(img_data)
    height_ = True
    ht, wd, cc = img.shape
    if (ht>=wd*1.333333):
        img=cv2.resize(img, (int(float(wd)*192/ht), 192), interpolation=cv2.INTER_LINEAR)
    else:
        img=cv2.resize(img, (144, int(float(ht)*144/wd)), interpolation=cv2.INTER_LINEAR)
        height_=False

    # create new image of desired size and color (white) for padding
    ww = 144
    hh = 192
    color = (255, 255, 255)
    result = np.full((hh, ww, cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - int(float(wd)*192/ht)) // 2
    yy = (hh-int(float(ht)*144/wd)) // 2
    # copy img image into center of result image
    if (height_):
        result[0:192, xx:img.shape[1]+xx] = img
    else:
        result[yy:img.shape[0]+yy, 0:144] = img



    im = Image.fromarray(result)
    im.save(str(index)+".png")


    index+=1