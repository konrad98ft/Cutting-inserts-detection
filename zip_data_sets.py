from zipfile import ZipFile
import os
from os.path import basename


PATHS=(r'C:\Users\Konrad\cutting-inserts-detection-master\traning',
       r'C:\Users\Konrad\cutting-inserts-detection-master\validation',
       r'C:\Users\Konrad\cutting-inserts-detection-master\test')

for dir in PATHS:
    # create a ZipFile object
    with ZipFile('traning.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(dir):
            for filename in filenames:
                #create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                print(filePath)
                zipObj.write(filePath, basename(filePath))