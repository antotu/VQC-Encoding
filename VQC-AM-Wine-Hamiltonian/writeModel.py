import os

numLayersList = [2, 4, 6, 8, 10]
ReuploadingList = ["False", "True"]


for numReps in range(10):
    for layer in numLayersList:
        for r in ReuploadingList:
            os.system(f"python3 WineClassification.py {layer} {r}")