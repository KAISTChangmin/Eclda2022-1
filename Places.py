import csv, matplotlib.pyplot as plt

openfilename = "csv/place_jeju.csv"
savefilename = "images/place_jeju.png"

#csv 파일 읽어서 list로 변환
f = open(openfilename, 'r', encoding='cp949')
rdr = csv.reader(f)
citylist = []
for line in rdr:
    citylist.append(line)
f.close()
citylist[0][0] = 0
citylist = list(map(lambda x: [int(x[0]), x[1], float(x[2]), float(x[3])], citylist))
size = len(citylist)

#각 City의 위치에 점을 찍고 그림을 그려서 파일로 저장
xlist = list(map(lambda x: x[2], citylist))
ylist = list(map(lambda x: x[3], citylist))
annotations = list(range(size))
plt.suptitle("Location of places in csv file", size=16, ha='center')
plt.scatter(xlist, ylist, color='black', alpha=0.8, edgecolor=None)
for j, label in enumerate(annotations):
    plt.annotate(label, (xlist[j], ylist[j]))
plt.gca().set_aspect('equal')
plt.savefig(savefilename)
plt.show()