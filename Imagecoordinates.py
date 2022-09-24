import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv

imagefilename = 'images/place_jeju.jpg'
savefilename = 'csv/place_jeju.csv'

#저장할 csv 파일 생성
f = open(savefilename,'w', newline='')
wr = csv.writer(f)

x, y, count = 0, 0, 0

#화면을 클릭하면 클릭한 점의 id와 좌표를 csv 파일에 저장 
def click(event):
    global x, y, count
    x = event.xdata
    y = event.ydata
    plt.plot(x, y, 'r*', markersize=5)
    wr.writerow([count, '', x, height - 1 - y])
    count += 1
    plt.show()

fig = plt.figure()
fig.canvas.mpl_connect("button_press_event", click)

#이미지 파일을 읽어서 열기
image_pil = Image.open(imagefilename)
image = np.array(image_pil)
height = len(image)
plt.imshow(image)
plt.show()
f.close()