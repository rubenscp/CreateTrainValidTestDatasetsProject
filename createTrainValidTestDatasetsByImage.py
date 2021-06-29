"""
Project: Creates the the train, valid and test datasets to use in the Yolo neural network.
Author: Rubens de Castro Pereira
Advisor: Dibio Leandro Borges
Date: 18/01/2021
Version: 1.0.0
"""

# Importing needed libraries
import os
import pathlib
import shutil
from random import randrange
import glob
import fnmatch
from datetime import datetime

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'


# ###########################################
# Application Methods
# ###########################################


# ###########################################
# Methods of Level 1
# ###########################################

# process images
def processImages(croppedImagesPath, trainValidTestDatasetsPath,
                  percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages):
    # creating train, valid and test directories
    createTrainValidTestDirectories(trainValidTestDatasetsPath)

    # process images of the classes
    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'adulta')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'instar1')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'instar2')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'instar3')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'instar4')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'exuvia')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'ovo')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'instar1ou2')

    organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages,
                              'instar3ou4')

    # defining counters
    # totalOfImages = 0
    # totalOfBoundingBoxes = 0
    # totalOfExuviaBoundingBoxesImages = 0
    # totalOfInstar1BoundingBoxesImages = 0
    # totalOfInstar2BoundingBoxesImages = 0
    # totalOfInstar3BoundingBoxesImages = 0
    # totalOfInstar4BoundingBoxesImages = 0
    # totalOfAdultaBoundingBoxesImages = 0
    # totalOfOvoBoundingBoxesImages = 0

    # calculating statistics
    # get the total number of images by class
    numberOfTrainImages = len(list(pathlib.Path(trainValidTestDatasetsPath + 'train/').glob('*.jpg')))
    numberOfValidImages = len(list(pathlib.Path(trainValidTestDatasetsPath + 'valid/').glob('*.jpg')))
    numberOfTestImages = len(list(pathlib.Path(trainValidTestDatasetsPath + 'test/').glob('*.jpg')))
    totalOfImages = numberOfTrainImages + numberOfValidImages + numberOfTestImages

    # printing statistics
    print('')
    print('Processing Statistics by Image:')
    print('-------------------------------')
    print('Date:', datetime.now())
    print('Total of train images : ', numberOfTrainImages)
    print('Total of valid images : ', numberOfValidImages)
    print('Total of test images  : ', numberOfTestImages)
    print('Total of images       : ', totalOfImages)


# ###########################################
# Methods of Level 2
# ###########################################

# create train, valid and test directories
def createTrainValidTestDirectories(trainValidTestDatasetsPath):
    if not os.path.exists(trainValidTestDatasetsPath):
        os.makedirs(trainValidTestDatasetsPath)

    directory = trainValidTestDatasetsPath + 'train'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = trainValidTestDatasetsPath + 'valid'
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = trainValidTestDatasetsPath + 'test'
    if not os.path.exists(directory):
        os.makedirs(directory)


# organize images by classes
def organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages, className):
    # setting the full path name
    croppedImagesClassNamePath = croppedImagesPath + className + '/'

    # get the total number of images by class
    numberOfTotalImages = len(list(pathlib.Path(croppedImagesClassNamePath).glob('*center*.jpg')))
    images = list(pathlib.Path(croppedImagesClassNamePath).glob('*center*.jpg'))

    # calculating the number of images used in train, valid and test datasets
    numberOfTrainImages = round(numberOfTotalImages * percentageOfTrainImages / 100.0)
    numberOfValidImages = round(numberOfTotalImages * percentageOfValidImages / 100.0)
    numberOfTestImages = numberOfTotalImages - numberOfTrainImages - numberOfValidImages

    # printing message in the console
    print(' ')
    print('Processing the class: ', className)
    print('Train: ', numberOfTrainImages)
    print('Valid: ', numberOfValidImages)
    print('Test: ', numberOfTestImages)
    print('Total: ', numberOfTotalImages)

    # moving to specific folders
    moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfTrainImages, 'train/')
    moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfValidImages, 'valid/')
    moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfTestImages, 'test/')


# ###########################################
# Methods of Level 3
# ###########################################

# move images and annotations files
def moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfImages,
                                specificDestinationFolder):
    # defining auxiliary variables
    imagesCounter = 0

    # process images
    while (imagesCounter < numberOfImages):
        # getting the files list
        # filesList2 = os.listdir(croppedImagesClassNamePath)
        # filesList3 = list(pathlib.Path(croppedImagesClassNamePath).glob('*center*.jpg'))
        # filesList4 = glob.glob(croppedImagesClassNamePath + '*center*.jpg')
        filesList = fnmatch.filter(os.listdir(croppedImagesClassNamePath), "*center*.jpg")

        # getting the random position
        index = randrange(len(filesList))

        # getting the file name
        fileName = filesList[index]

        # check if file is an image or not
        if fileName.lower().find('jpg') == -1 and fileName.lower().find('jpeg') == -1:
            continue

        # moving image and annotation of center position
        centerImageName = getImageFileNameWithouExtension(fileName)
        moveImageAndAnnotationFile(croppedImagesClassNamePath, centerImageName,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        # get position of string 'center.jpg'
        centerStringPosition = fileName.find('center')

        # moving image and annotation of another positions
        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'north'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'south'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'east'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'west'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'northeast'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'northwest'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'southeast'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        imageNameOfAnotherPosition = fileName[0:centerStringPosition] + 'southwest'
        moveImageAndAnnotationFile(croppedImagesClassNamePath, imageNameOfAnotherPosition,
                                   trainValidTestDatasetsPath, specificDestinationFolder)

        # # move image file
        # source = croppedImagesClassNamePath + fileName
        # destination = trainValidTestDatasetsPath + specificDestinationFolder + fileName
        # shutil.move(source, destination)
        #
        # # move annotation file
        # source = croppedImagesClassNamePath + getImageFileNameWithouExtension(fileName) + '.txt'
        # destination = trainValidTestDatasetsPath + specificDestinationFolder + getImageFileNameWithouExtension(
        #     fileName) + '.txt'
        # shutil.move(source, destination)

        # # saving processing results
        # saveProcessingResults(trainValidTestDatasetsPath, fileName)
        #
        # counting files moved
        imagesCounter += 1


# ###########################################
# Methods of Level 4
# ###########################################


def moveImageAndAnnotationFile(croppedImagesClassNamePath, fileName, trainValidTestDatasetsPath,
                               specificDestinationFolder):
    # defining the image name source
    source = croppedImagesClassNamePath + fileName + '.jpg'

    # checking if  image file exists
    if not os.path.isfile(source):
        return

    # move image file
    destination = trainValidTestDatasetsPath + specificDestinationFolder + fileName + '.jpg'
    shutil.move(source, destination)

    # move annotation file
    source = croppedImagesClassNamePath + fileName + '.txt'
    destination = trainValidTestDatasetsPath + specificDestinationFolder + fileName + '.txt'
    shutil.move(source, destination)

    # saving processing results
    saveProcessingResults(trainValidTestDatasetsPath, specificDestinationFolder, fileName)


# ###########################################
# Methods of Level 5
# ###########################################


# get image filiename without extension name
def getImageFileNameWithouExtension(fileName):
    # getting jpg position
    jpegPosition = -1
    jpegPosition = fileName.find('jpg')
    if jpegPosition == -1: jpegPosition = fileName.find('jpeg')
    if jpegPosition == -1: jpegPosition = fileName.find('JPG')
    if jpegPosition == -1: jpegPosition = fileName.find('JPEG')

    # getting only image name
    imageFileName = fileName[:jpegPosition - 1]

    # returning image file name
    return imageFileName


# save results of processing
def saveProcessingResults(trainValidTestDatasetsPath, specificDestinationFolder, fileName):
    # creating the processing results file
    processingResultsFile = open(trainValidTestDatasetsPath + 'processingResults.txt', 'a+')

    # replacing characters to split string
    fileName = fileName.replace('-', '.')

    # getting the array of names parts
    values = fileName.split('.')

    # setting line to write
    line = specificDestinationFolder + ' ' \
           + values[0] + ' ' \
           + values[1] + ' ' \
           + values[3] + ' ' \
           + values[4] + ' ' \
           + LINE_FEED

    # write line
    processingResultsFile.write(line)

    # closing file
    processingResultsFile.close()


# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    # INPUT_CROPPED_BOUNDING_BOXES_DATABASE_PATH = \
    #     'E:/desenvolvimento/projetos/DoctoralProjects/CreateTrainValidTestDatasetsProjectImages/Block 30/30.2 White Fly Cropped Images by Classes/'
    # OUTPUT_TRAIN_VALID_TEST_DATASET_PATH = \
    #     'E:/desenvolvimento/projetos/DoctoralProjects/CreateTrainValidTestDatasetsProjectImages/Block 30/30.3 White Fly Cropped Images by Train-Valid-Test/'

    INPUT_CROPPED_BOUNDING_BOXES_DATABASE_PATH = \
        'E:/desenvolvimento/projetos/DoctoralProjects/WhiteFlyExperiment/01.06 - Training - Cropped Images by Classes (128x128 pixels)/'
    OUTPUT_TRAIN_VALID_TEST_DATASET_PATH = \
        'E:/desenvolvimento/projetos/DoctoralProjects/WhiteFlyExperiment/01.07 - Training - Train-Valid-Test Cropped Images (128x128 pixels)/'

    print('Organize cropped images into train, valid and test folders')
    print('---------------------------------')
    print('')
    print('Input images path  : ', INPUT_CROPPED_BOUNDING_BOXES_DATABASE_PATH)
    print('Output images path : ', OUTPUT_TRAIN_VALID_TEST_DATASET_PATH)
    print('')

    # setting the percentual of each dataset
    percentageOfTrainImages = 70
    percentageOfValidImages = 20
    percentageOfTestImages = 10

    # processing the annotated images
    processImages(INPUT_CROPPED_BOUNDING_BOXES_DATABASE_PATH, OUTPUT_TRAIN_VALID_TEST_DATASET_PATH,
                  percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages)

    # end of processing
    print()
    print('End of processing')
