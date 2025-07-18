import os

EncodingList =EncodingList =["X", "XY", "XYZ", "XZ", "XZY", "Y", "YX", "YXZ", "YZ", "YZX",
               "Y_H", "YX_H", "YXZ_H", "YZ_H", "YZX_H", "Z_H", "ZX_H", "ZXY_H", "ZY_H", "ZYX_H"]

numLayersList = [2, 4, 6, 8, 10]
ReuploadingList = ["False", "True"]


for numReps in range(10):
    for enc in EncodingList:
        
        for layer in numLayersList:
            for r in ReuploadingList:
                print(enc, layer, r)
                if "_H" in enc:
                    enc_2 = enc.split("_")[0]
                    os.system(f"python DiabetesClassification.py {enc_2} {layer} True {r}")
                else:
                    os.system(f"python DiabetesClassification.py {enc} {layer} False {r}")