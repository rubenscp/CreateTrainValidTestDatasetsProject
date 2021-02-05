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

# from re import _expand

# import cv2
#
# from Entity.BoundingBox import BoundingBox
# from Entity.Image import Image
# from Entity.Pixel import Pixel
# from Entity.GroundTruthData import GroundTruthData
# from Entity.DetectedObject import DetectedObject

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
CROPPED_BOUNDING_BOXES_DATABASE_PATH = 'E:/desenvolvimento/projetos/DoctoralProjects/Images-Input-Output/02. White Fly - Cropped Bounding Boxes Images by Classes/'
TRAIN_VALID_TEST_DATASET_PATH = 'E:/desenvolvimento/projetos/DoctoralProjects/Images-Input-Output/03. White Fly - Train-Valid-Test Datasets/'


# ###########################################
# Application Methods
# ###########################################

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


# move images and annotations files
def moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfImages,
                                specificDestinationFolder):
    # defining auxiliary variables
    imagesCounter = 0

    # process images
    while (imagesCounter < numberOfImages):
        # getting the files list
        filesList = os.listdir(croppedImagesClassNamePath)

        # getting the random position
        index = randrange(len(filesList))

        # getting the file name
        fileName = filesList[index]

        # check if file is an image or not
        if fileName.lower().find('jpg') == -1 and fileName.lower().find('jpeg') == -1:
            continue

        # move image file
        source = croppedImagesClassNamePath + fileName
        destination = trainValidTestDatasetsPath + specificDestinationFolder + fileName
        shutil.move(source, destination)

        # move annotation file
        source = croppedImagesClassNamePath + getImageFileNameWithouExtension(fileName) + '.txt'
        destination = trainValidTestDatasetsPath + specificDestinationFolder + getImageFileNameWithouExtension(
            fileName) + '.txt'
        shutil.move(source, destination)

        # saving processing results
        saveProcessingResults(trainValidTestDatasetsPath, fileName)

        # counting files moved
        imagesCounter += 1


# process images
def organizeImagesByClassName(croppedImagesPath, trainValidTestDatasetsPath,
                              percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages, className):
    # setting the full path name
    croppedImagesClassNamePath = croppedImagesPath + className + '/'

    # get the total number of images by class
    numberOfTotalImages = len(list(pathlib.Path(croppedImagesClassNamePath).glob('*.jpg')))
    images = list(pathlib.Path(croppedImagesClassNamePath).glob('*.jpg'))

    # calculating the number of images used in train, valid and test datasets
    numberOfTrainImages = round(numberOfTotalImages * percentageOfTrainImages / 100.0)
    numberOfValidImages = round(numberOfTotalImages * percentageOfValidImages / 100.0)
    numberOfTestImages = numberOfTotalImages - numberOfTrainImages - numberOfValidImages

    # moving to specific folders
    moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfTrainImages, 'train/')
    moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfValidImages, 'valid/')
    moveImageAndAnnotationFiles(croppedImagesClassNamePath, trainValidTestDatasetsPath, numberOfTestImages, 'test/')


# save results of processing
def saveProcessingResults(trainValidTestDatasetsPath, fileName):
    # creating the processing results file
    processingResultsFile = open(trainValidTestDatasetsPath + 'processingResults.txt', 'a+')

    # replacing characters to split string
    fileName = fileName.replace('-', '.')

    # getting the array of names parts
    values = fileName.split('.')

    # setting line to write
    line = values[0] + ' ' \
           + values[1] + ' ' \
           + values[3] + ' ' \
           + values[4] + ' ' \
           + LINE_FEED

    # write line
    processingResultsFile.write(line)

    # closing file
    processingResultsFile.close()


# process images
def processImages(croppedImagesPath, trainValidTestDatasetsPath,
                  percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages):
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

    # printing statistics
    # print('')
    # print('Estatísticas do Processamento:')
    # print('------------------------------')
    # print('Total de imagens             : ', totalOfImages)
    # print('Total de bounding boxes      : ', totalOfBoundingBoxes)
    # print('Total de imagens de exuvia   : ', totalOfExuviaBoundingBoxesImages)
    # print('Total de imagens de instar1  : ', totalOfInstar1BoundingBoxesImages)
    # print('Total de imagens de instar2  : ', totalOfInstar2BoundingBoxesImages)
    # print('Total de imagens de instar3  : ', totalOfInstar3BoundingBoxesImages)
    # print('Total de imagens de instar4  : ', totalOfInstar4BoundingBoxesImages)
    # print('Total de imagens de adultas  : ', totalOfAdultaBoundingBoxesImages)
    # print('Total de imagens de ovo      : ', totalOfOvoBoundingBoxesImages)
    # print('Máximo Height                : ', sizeSquareImage)
    # print('Máximo Width                 : ', sizeSquareImage)
    # print('')


# ###########################################
# Main method
# ###########################################
if __name__ == '__main__':
    print('Cropping Annotated Bounding Boxes')
    print('---------------------------------')
    print('')
    print('Input images path - Cropped images path  : ', CROPPED_BOUNDING_BOXES_DATABASE_PATH)
    print('Output images path                       : ', TRAIN_VALID_TEST_DATASET_PATH)
    print('')

    # setting the percentual of each dataset
    percentageOfTrainImages = 70
    percentageOfValidImages = 20
    percentageOfTestImages = 10

    # processing the annotated images
    processImages(CROPPED_BOUNDING_BOXES_DATABASE_PATH, TRAIN_VALID_TEST_DATASET_PATH,
                  percentageOfTrainImages, percentageOfValidImages, percentageOfTestImages)

    # end of processing
    print('End of processing')
