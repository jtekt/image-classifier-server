from tensorflow.python.client import device_lib
import os
import shutil

def getGpus():
    devices = device_lib.list_local_devices()
    gpus = [d for d in devices if d.device_type == "GPU"]
    return gpus

def lookDeeperIfNeeded(parentDirectory):
    
    folderContent = next(os.walk(parentDirectory))[1]

    # If there is a folder other than "variables", need to look into it
    if folderContent:
        for i in range(len(folderContent)):
            if (not folderContent[i] == 'variables'):
                innerFolderPath = os.path.join(parentDirectory, folderContent[i])
                innerFolderContent = next(os.walk(innerFolderPath))[1]
                innerFileContent = next(os.walk(innerFolderPath))[2]

                # If multiple files, move them up one folder
                for subSubFile in innerFileContent:
                    subSubFilePath = os.path.join(innerFolderPath, subSubFile)
                    if os.path.exists(os.path.join(parentDirectory, subSubFile)):
                        os.remove(os.path.join(parentDirectory, subSubFile))
                    shutil.move(subSubFilePath, parentDirectory)
                
                # If multiple folders, move them up one folder
                for subSubFolder in innerFolderContent:
                    subSubFolderPath = os.path.join(innerFolderPath, subSubFolder)
                    if os.path.exists(os.path.join(parentDirectory, subSubFolder)):
                        shutil.rmtree(os.path.join(parentDirectory, subSubFolder), ignore_errors=True)
                    shutil.move(subSubFolderPath, parentDirectory)
                
                shutil.rmtree(innerFolderPath, ignore_errors=True)
