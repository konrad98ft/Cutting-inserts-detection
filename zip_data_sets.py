from zipfile import ZipFile
import os
from os.path import basename

'''
Script used to zip files from traininf test and validation directories.
Use split.py to split images into mentdioned directories.
Those files should be put in 1 directory and upload to Google Colab environment.
'''

# Pathes to the particular folders
PATHS=('\traning',
       '\validation',
       '\test')

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