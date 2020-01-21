import os
import time
from _datetime import datetime

import pandas as pd
from tqdm import tqdm

FORMAT = 64
DATA_DIR = f"..\\train_simplified_images\\{64}"
SAVE_FILE = "data\\data"
STEP = 10000

dirs = os.listdir(DATA_DIR)
i = 0
num = int(i / STEP) + 1
print(f"\nstarted {num}/{10}")
time.sleep(0.01)
df = pd.DataFrame()

for index, word in enumerate(tqdm(dirs)):
	directory = DATA_DIR + "\\" + word + "\\" + word + ".csv"
	single_df = pd.read_csv(directory)[i:i+1000]
	# single_df = pd.read_csv(directory)
	df = df.append(single_df, ignore_index=True)
	del single_df
save_file_name = f"{SAVE_FILE}_{num}.csv"
df.to_csv(save_file_name, index=False)
print(f"saved {save_file_name}\nfinished {datetime.now()}\n")
