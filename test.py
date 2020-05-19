import glob
import os


for file in glob.glob(os.path.join("H:\\PycharmProjects\\dataset\\data_thchs30\\train", "*.wav")):
    print(os.path.splitext(os.path.split(file)[-1])[0])