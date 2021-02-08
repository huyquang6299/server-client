import cv2


def ImageCut(img):
    # Open slot coordinate file, read in
    f2 = open('Slot coordinate.txt', 'r')
    lis = [line.split() for line in f2]
    # print(lis)
    coord = []
    n = len(lis[0])
    fileLength = len(lis)
    f2.close()
    f2 = open('Slot coordinate.txt', 'r')
    countLine = len(f2.read().split('\n')) - 1
    f2.close()
    for j in range(n):
        tempArray = []
        for i in range(fileLength):
            # print(lis[i][j])
            tempArray.append(int(lis[i][j]))
        coord.append(tempArray)
        # print(tempArray)
    fSlot = open('Slot coordinate.txt')
    text = fSlot.read().split('\n')[:-1]
    coordition = []
    for line in text:
        coordArray = []
        for coord2 in line.split(' '):
            coordArray.append(int(coord2))

        coordition.append(coordArray)
    fSlot.close()

    imgArray = []
    for index in range(0, countLine):
        x1 = coordition[index][1]
        y1 = coordition[index][2]
        x2 = coordition[index][3]
        y2 = coordition[index][4]
        imgCut = img[y1:y2,x1:x2]
        imgArray.append(imgCut)

    return imgArray, countLine, coordition
