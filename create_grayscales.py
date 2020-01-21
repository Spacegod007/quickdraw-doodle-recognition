import ast
import os
import time
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from glob import glob
from tqdm import tqdm

DATA_DIR = "train_simplified"
# SAVE_DIR = f"..\\{DATA_DIR}_images"
SAVE_DIR = r"D:\Users\Jordi\image data"
OUTPUT_SIZE = 32, 32
DATA_COLUMN = "image_data"
WORD_COLUMN = "word"
SAMPLE_SIZE = 10_000

RESUME = 306

files = glob(DATA_DIR + "\\" + "*.csv")
for index, file in enumerate(files[RESUME:RESUME+34]):
    time.sleep(0.01)
    print(f"{os.linesep + os.linesep if not index == 0 else os.linesep}{file} - {RESUME + index + 1}/{len(files)}")
    time.sleep(0.01)
    d = {WORD_COLUMN: []}
    for i in range(OUTPUT_SIZE[0]*OUTPUT_SIZE[1]):
        d[i] = []
    data = pd.read_csv(file)
    data = data.loc[data['recognized']]
    data = data[:SAMPLE_SIZE]
    word = file.split('\\', 2)[1].split('.', 2)[0]

    directory = SAVE_DIR + "\\" + word
    if not os.path.exists(directory):
        os.makedirs(directory)

    drawings = [ast.literal_eval(pts) for pts in data.drawing.values]
    del data

    for drawing_index, drawing in enumerate(tqdm(drawings)):
        fig = plt.figure(figsize=(1, 1))
        for x, y in drawing:
            plt.plot(x, y, color='black')
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('L')
        buf.close()
        img = img.resize(OUTPUT_SIZE, Image.ANTIALIAS)
        img.save(f"{directory}\\{word}_{drawing_index}.png")

        # array = np.array(img.getdata())
        # for array_index, value in enumerate(array):
        #     d[array_index].append(value)
        # d[WORD_COLUMN].append(word)

    # single_df = pd.DataFrame(d)
    # single_df.to_csv(f"{directory}\\{word}.csv", index=False)
    # del single_df
