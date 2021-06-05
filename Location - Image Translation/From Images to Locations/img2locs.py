import os
import glob
import math
import numpy as np
from PIL import Image

grid_unit = 0.0002
sg_x_min = 103.61
sg_x_max = 104.023
sg_y_min = 1.228
sg_y_max = 1.476
mRow = int((sg_y_max - sg_y_min) * 1.0 / grid_unit)
nCol = int((sg_x_max - sg_x_min) * 1.0 / grid_unit)


def _location_to_image():
    usefulPath = "the\\path\\where\\your\\data\\files\\were\\stored"
    for folderName in os.listdir(usefulPath):
        dataPath = usefulPath + folderName + "\\"
        for filename in glob.glob(os.path.join(dataPath, 'suffix_of_your_file_name')):
            imgarr = np.zeros([mRow, nCol, 3], dtype=np.uint8)
            lines = open(filename).read().split('\n')
            fileLength = (len(lines))
            for i in range(fileLength - 1):
                line = lines[i].split(',')
                if sg_x_min < float(line[1]) < sg_x_max and sg_y_min < float(line[2]) < sg_y_max:
                    x = math.ceil((float(line[1]) - sg_x_min) / grid_unit)
                    y = math.ceil((float(line[2]) - sg_y_min) / grid_unit)
                    try:
                        imgarr[mRow - y, x - 1, :] = [255, 255, 255]

                        imgarr[mRow - y - 1, x - 1 - 1, :] = [255, 255, 255]
                        imgarr[mRow - y - 1, x - 1, :] = [255, 255, 255]
                        imgarr[mRow - y - 1, x, :] = [255, 255, 255]
                        imgarr[mRow - y, x - 1 - 1, :] = [255, 255, 255]
                        imgarr[mRow - y, x, :] = [255, 255, 255]
                        imgarr[mRow - y + 1, x - 1 - 1, :] = [255, 255, 255]
                        imgarr[mRow - y + 1, x - 1, :] = [255, 255, 255]
                        imgarr[mRow - y + 1, x, :] = [255, 255, 255]
                    except LookupError:
                        print("Index Error Exception Raised, list index out of range")
            imgarr = 255 - imgarr
            img = Image.fromarray(imgarr)
            picName = 'path\\where\\you\\want\\your\\generated\\files\\to\\be\\stored'
            img.save(picName)