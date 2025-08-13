import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image  # ??
from pylab import *  # Matplotlib 中的 PyLab 接口包含很多方便用户创建图像的函数
import numpy as np
import os

import copy
from PIL import Image
from pylab import *
from scipy.cluster.vq import *

# The folder that contains the ORing images
# directory = r'G:/Microsee/camera_calibration/CameraCalibration-master/src/images/calibrated/2/'
directory = r'./src/images/calibrated/2/'

# Imports and returns a list of all the files from the specified folder
def read_images(directory):
    image_list = []
    for file in os.listdir(directory):
        image_list.append([os.path.join(directory + '/', file), cv.imread(os.path.join(directory + '/', file), 0)])
    return image_list


# Stores histogram of pixel levels from retrieved image
def image_hist(image):
    hist = np.zeros(256)
    for i in range(0, image.shape[0]):  # Loops through the rows
        for j in range(0, image.shape[1]):  # Loops through the columns
            hist[image[i, j]] += 1  # Increases the pixel count for this level by 1
    return hist


# Returns a list of the two peak values from retrieved histogram
def hist_peaks(hist):
    # peaks = [np.where(hist == max(hist))[0][0]]
    # peak1 = peaks[0] - 100
    # peak2 = peaks[0] + 100
    # temp_array = [hist[i] for i in range(len(hist)) if i < peak1 or i > peak2]
    # peaks.append(temp_array.index(max(temp_array) ))
    # peaks.sort()
    peaks = [np.where(hist == max(hist))[0][0]]
    all_valley = [np.where(hist <= np.mean(hist))[0]][0]
    valley = all_valley[all_valley < peaks]
    # 找距离peaks最近的valley的索引值，即灰度值
    dist = (peaks[0] - valley)
    minInxInDist = np.where(dist == min(dist))[0]
    threshIdx = valley[minInxInDist]
    return threshIdx


# Calculates a threshold value for the retrieved image
def threshold_value(image):
    hist = image_hist(image)  # Retrieves histogram of pixel values for the given image
    tValue = hist_peaks(hist)[0]  # Retrieve the 2 peaks of pixel values in the histogram
    # tValue = peaks[0] + int((peaks[1] - peaks[0]) / 2) # Stores the threshold value from the valley of the 2 peaks
    return tValue


# Applies thresholding to the retrieved image and returns it
def threshold(image):
    tImage = image.copy()
    # tValue = threshold_value(image)
    # for x in range(0, tImage.shape[0]):
    #     for y in range(0, tImage.shape[1]):
    #         if tImage[x,y] > tValue:
    #             tImage[x,y] = 255
    #         else:
    #             tImage[x,y] = 0
    dy = tImage.shape[0]  # row
    dx = tImage.shape[1]
    features = []
    for y in range(dy):
        for x in range(dx):
            R = tImage[y, x, 0] / 255
            G = tImage[y, x, 1] / 255
            B = tImage[y, x, 2] / 255
            features.append([R, G, B, y / dy, x / dx])
    features = np.array(features, 'f')

    # Cluster.
    centroids, variance = kmeans(features, 2)
    code, distance = vq(features, centroids)
    code = code * 255
    codeim = code.reshape(dy, dx)
    codeim = Image.fromarray(codeim).resize(tImage.shape[:2]).convert('1')
    figure()
    imshow(codeim)
    show()
    return tImage


# Applies erosion to the retrieved image using the morphological structure and returns the new image
def erosion(image, struct):
    eImage = image.copy()
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            # Checks if the selected pixel matches the foreground colour and the index positions required for the morphological structure to be within the bounds of the image
            if image[i, j] == 0 and i - 1 >= 0 and j - 1 >= 0 and i + 1 < image.shape[0] and j + 1 < image.shape[1]:
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        offset_i = i + y
                        offset_j = j + x

                        # Checks if the position being checked is not the current pixel and that it aligns with the designation within the morphological structure, 1 in structure and 255 in image
                        if [offset_i, offset_j] != [i, j] and struct[y + 1][x + 1] == 1 and image[
                            offset_i, offset_j] == 255:
                            eImage[i, j] = 255  # Sets the current pixel to the background of the image
    return eImage


# Applies dilation to the retrieved image using the morphological structure and returns the new image
def dilation(image, struct):
    dImage = image.copy()
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            # Checks if the selected pixel matches the foreground colour and the index positions required for the morphological structure to be within the bounds of the image
            if image[i, j] == 255 and i - 1 >= 0 and j - 1 >= 0 and i + 1 < image.shape[0] and j + 1 < image.shape[1]:
                for y in range(-1, len(struct) - 1):
                    for x in range(-1, len(struct) - 1):
                        offset_i = i + y
                        offset_j = j + x

                        # Checks if the position being checked is not the current pixel and that it aligns with the designation within the morphological structure, 1 in structure and 0 in image
                        if [offset_i, offset_j] != [i, j] and struct[y + 1][x + 1] == 1 and image[
                            offset_i, offset_j] == 0:
                            dImage[i, j] = 0  # Sets the current pixel to the background of the image
    return dImage


# Retrieves an image and morphological structure, then applies closing which involves dilation followed by erosion and returns a new image
def closing(image, struct):
    cImage = dilation(image, struct)
    cImage = erosion(cImage, struct)
    return cImage


# Takes in an image and performs connected component labelling, returning the list of labels for pixels at every index position on the image
def component_label(image):
    # label_list = [[0 for j in range(0, image.shape[1])] for i in range(0, image.shape[0])] # Creates a 2D list with the size of an image and filled with 0s for unlabled attributes
    label_list = np.zeros((image.shape[0], image.shape[1]))
    curlab = 1  # Declares the current label tag to 1
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] == 0 and label_list[i][j] == 0:
                label_list[i][j] = curlab
                queue = []  # Initialise an empty list that will function as a queue structure
                queue.append([i, j])  # Adds the current pixels coordinates to the queue list

                # Loops while the length of the queue is more than 0 due to neighbouring coordinates still requiring processing
                while len(queue) > 0:
                    item = queue.pop(0)
                    if 0 < item[0] < image.shape[0] - 1 and 0 < item[1] < image.shape[1] - 1:
                        if image[item[0] - 1, item[1]] == 0 and label_list[item[0] - 1][item[
                            1]] == 0:  # Checks if the neighbour above is a foreground pixel and is currently unlabelled
                            queue.append([item[0] - 1, item[1]])  # Adds its coordinates to the queue
                            label_list[item[0] - 1][item[1]] = curlab  # Labels it with the current component label
                        if image[item[0] + 1, item[1]] == 0 and label_list[item[0] + 1][item[
                            1]] == 0:  # Checks if the neighbour below is a foreground pixel and is currently unlabelled
                            queue.append([item[0] + 1, item[1]])
                            label_list[item[0] + 1][item[1]] = curlab
                        if image[item[0], item[1] - 1] == 0 and label_list[item[0]][item[
                                                                                        1] - 1] == 0:  # Checks if the neighbour to the left is a foreground pixel and is currently unlabelled
                            queue.append([item[0], item[1] - 1])
                            label_list[item[0]][item[1] - 1] = curlab
                        if image[item[0], item[1] + 1] == 0 and label_list[item[0]][item[
                                                                                        1] + 1] == 0:  # Checks if the neighbour to the right is a foreground pixel and is currently unlabelled
                            queue.append([item[0], item[1] + 1])
                            label_list[item[0]][item[1] + 1] = curlab
                curlab += 1
    return label_list


# Retrives the list of labels, a label for calculating the area and returns the number of pixels in the area with the label
def calculate_area(label_list, curlab):
    area = 0
    for x in range(0, len(label_list)):
        for y in range(0, len(label_list[0])):
            if label_list[x][
                y] == curlab:  # Checks if the current label at this index position matches the one we are trying to count
                area += 1
    return area


# Retrieves an image and the list of labels then returns the new image with painted labels
def paint_labels(image, label_list):
    pImage = image.copy()
    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            pImage[x, y] = label_list[x][
                               y] * 255  # Updates the labelled pixel by multiplying it by 255 which sets the foreground colours to white and background to black
    return pImage


# Retrieves a list of labels and removes the smallest areas
def remove_smallest_areas(label_list):
    unique_labels = np.unique(label_list)  # Extracts the list of unique component labels
    if len(unique_labels) > 2:
        unique_labels = unique_labels[1:]
        areas = []
        for i in range(len(unique_labels)):
            areas.append(calculate_area(label_list, unique_labels[
                i]))  # Calls a method to calculate the area of the currently selected label/component
        # Removing the smallest area
        smallest_area = unique_labels[areas.index(min(areas))]
        new_labels = []
        for label_set in label_list:
            new_set = []
            for label in label_set:
                if label == smallest_area:
                    new_set.append(0)
                else:
                    new_set.append(label)
            new_labels.append(new_set)
        return new_labels  # Returns the modified labels list
    return label_list


# Retrieves a list of labels and returns the centroid coordinates by calculating the average i and the averagre j index positions for the oring
def get_centroid(image):
    pixel = 0
    total_i = 0
    total_j = 0
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if image[i][j] == 1:
                pixel += 1
                total_j += j
                total_i += i
    return [int(total_i / pixel),
            int(total_j / pixel)]  # The sum of foreground i position divided by the count of foreground pixels gives the average i, likewise for j


# Retrieves a list of labels and calculates the coordinates for the top left and bottom right corners of the bounding box
def make_bounding_box(image):
    bound_coords = [0 for i in range(4)]
    first_pixel = False  # A failsafe measure for insuring that min/min coordinate initilisation does not happen before a foreground pixel is selected
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if image[i][j] == 1:
                if first_pixel == True:  # Checks if this is the first relevant pixel to be processed
                    if i < bound_coords[0]:
                        bound_coords[0] = i
                    elif i > bound_coords[1]:
                        bound_coords[1] = i
                    if j < bound_coords[2]:
                        bound_coords[2] = j
                    elif j > bound_coords[3]:
                        bound_coords[3] = j
                else:  # Else if it is the first pixel found, it will set both the min and the max values for i and j with the current coordinates
                    first_pixel = True
                    bound_coords[0] = i
                    bound_coords[1] = i
                    bound_coords[2] = j
                    bound_coords[3] = j
    return bound_coords


# Retrieves an image, bounding box doorindates, a boolean result and returns the image with a bounding box around the ORing
def draw_bounding_box(image, bounding_box, result):
    pixel = (0, 0, 255)
    if result == True:
        pixel = colour = (0, 255, 0)

    # Draws the lines of the bounding box using the min and max of i and j coordinates
    image[(bounding_box[0] - 1):(bounding_box[1] + 2), bounding_box[2] - 1] = pixel
    image[(bounding_box[0] - 1):(bounding_box[1] + 2), bounding_box[3] + 1] = pixel
    image[bounding_box[0] - 1, (bounding_box[2] - 1):(bounding_box[3] + 2)] = pixel
    image[bounding_box[1] + 1, (bounding_box[2] - 1):(bounding_box[3] + 2)] = pixel
    return image


# Retrieves an image and centroid coordinate, then returns the average inne and outer edge radius values as a list with inner being 0 and outer being 1
def calculate_radius(image, centroid):
    radius = [0, 0]
    found = [False,
             False]  # Used for differentiating between the inner and otuer edge radius values and when to stop looking for them

    # Centroid going up
    for i in range(centroid[0] - 1, 0, -1):  # Loops from the centroid y position to the top of the image
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        if image[i, centroid[1]] == 255 and image[i + 1, centroid[
            1]] == 0:  # Checks if the current pixel is of a foregorund colour and the previous pixel was background, then the inner edge has been found
            found[0] = True
        if image[i, centroid[1]] == 0 and image[i + 1, centroid[
            1]] == 255:  # Checks if the current pixel is of a background colour and the previous pixel was foreground, then the outer edge has been reached
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break  # Breaks out of the loop after both edges have been found

    # The following 3 loops repeat this cycle as above but going in different directions

    # Centroid going down
    for i in range(centroid[0], 220):
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        if image[i, centroid[1]] == 255 and image[i - 1, centroid[1]] == 0:
            found[0] = True
        if image[i, centroid[1]] == 0 and image[i - 1, centroid[1]] == 255:
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    # Centroid going left
    for j in range(centroid[1], 0, -1):
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        if image[centroid[0], j] == 255 and image[centroid[0], j + 1] == 0:
            found[0] = True
        if image[centroid[0], j] == 0 and image[centroid[0], j + 1] == 255:
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    # Centroid going right
    for j in range(centroid[1], 220):
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        if image[centroid[0], j] == 255 and image[centroid[0], j - 1] == 0:
            found[0] = True
        if image[centroid[0], j] == 0 and image[centroid[0], j - 1] == 255:
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    radius = [num / 4 for num in radius]
    radius = [num - 1 for num in radius]
    return radius


# Retrieves x and y pixel coordinates to check, a and b coordinates for the center point of the circle and the radius, then returns true if x and y coordinates are inside the circle
def circle_center(x, y, a, b, radius):
    return ((x - a) ** 2) + ((y - b) ** 2) <= radius ** 2


# Retrieves an image, centroid coordinates, radius values and returns whether the ORing passes or fails
def oring_result(image, centroid, radius):
    allowed_diff = 2  # The distance allowed between a foreground pixel and the constructed ring to still pass
    shape_pixel = 255  # The pixel value assigned to the shape of the constructed ring which is used for processing
    shape = image.copy()

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            # Calls a method which calculates if the pixel locations i and j are within the expected ring
            if circle_center(j, i, centroid[1], centroid[0], radius[1]) and not circle_center(j, i, centroid[1],
                                                                                              centroid[0], radius[0]):
                shape[i, j] = shape_pixel
            else:
                shape[i, j] = 0

    # A list that stores any faulty pixel coordinates
    faulty_coords = []

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            # Checks if the background pixel is located where the ring should be or if a foreground pixel is found outside the ring
            if image[i, j] == 0 and shape[i, j] == shape_pixel or image[i, j] == 255 and shape[i, j] == 0:
                pixel_faulty = True
                # Loops from -2 x,y coordinates to +2 x,y from the current pixel location
                for x in range(-allowed_diff, allowed_diff + 1):
                    for y in range(-allowed_diff, allowed_diff + 1):
                        if [x, y] != [0, 0]:
                            offset_x = i + x
                            offset_y = j + y
                            if image[offset_x, offset_y] == 255:
                                pixel_faulty = False

                if pixel_faulty:
                    faulty_coords.append([i,
                                          j])  # Adds its coordinates to the faulty pixel coordinates list for recolouring the faulty area

    if len(faulty_coords) > 0:
        return [False, faulty_coords, shape]
    return [True, faulty_coords, shape]


# Retrieves an image and the expected ring shape
def draw_faulty_locations(image, expected_ring):
    colour = np.array([0, 255, 0])  # Sets the drawing colour as green for the fix areas
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            # Checks if a foreground pixel is outside the expected ring or if a background pixel is inside the expected ring
            if np.array_equal(expected_ring[i, j], np.array([255, 255, 255])) and np.array_equal(image[i, j], np.array(
                    [0, 0, 0])) or np.array_equal(expected_ring[i, j], np.array([0, 0, 0])) and np.array_equal(
                    image[i, j], np.array([255, 255, 255])):
                image[i, j] = colour  # Draws the pixel as green
    return image


# Retrieves the title for the file name, the image, labels, centroid coords, bounding box coordinates around the ring, the processing time from the start as well as the total amount of passes and fails for the ORings
def start_process(title, image, labels, centroid, bounding_box, start, totals):
    drawn_image = image.copy()  # Creates a copy of the original image to draw the fixed pixels
    drawn_image = paint_labels(drawn_image,
                               labels)  # Paints the labels of the components on the image which inverts the foreground and background colours
    radius = calculate_radius(drawn_image,
                              centroid)  # Calculates the 2 radius values which are the inner and outer edge of the ORing
    result = oring_result(drawn_image, centroid,
                          radius)  # Processes the image and returns true or false for whether it passed
    finish = time.time()  # Retrieves the time for when all the processing is complete
    results[2] = round((finish - start),
                       3)  # Adds the processing time for the current ring to the total processing time count rounded to 3 decimals
    drawn_image = cv.cvtColor(drawn_image,
                              cv.COLOR_GRAY2RGB)  # Converts the image into BGR scale which allows for the use of colour rather than greyscale
    result[2] = cv.cvtColor(result[2],
                            cv.COLOR_GRAY2RGB)  # Converts the drawn pixels into BGR scale which allows to colour the fixed pixels

    # Checks if the result for the current ORing has come back as true or false
    if result[0] == False:
        result_output = 'FAILED'
        colour = (0, 0, 255)
        results[1] += 1
        drawn_image = draw_faulty_locations(drawn_image, result[
            2])  # Draws in the areas that are expected but missing in green within the circle
    else:
        result_output = 'PASSED'
        colour = (0, 255, 0)
        results[0] += 1

    # Declares a font and line type variable
    font = cv.FONT_HERSHEY_SIMPLEX
    line = cv.LINE_AA

    drawn_image = draw_bounding_box(drawn_image, bounding_box, result[0])  # Draws the bounding box around the circle
    drawn_image = cv.putText(drawn_image, "Time: " + str(round(finish - start, 3)), (5, 10), font, 0.4, (255, 255, 255),
                             1, line)  # Paints the processing time text onto the image
    drawn_image = cv.putText(drawn_image, result_output, (170, 10), font, 0.4, colour, 1,
                             line)  # Prints the process result top right of the interface
    drawn_image = cv.putText(drawn_image, "Results: ", (5, 210), font, 0.4, (255, 255, 255), 1,
                             line)  # Prints results beside passed or failed for the results of tests
    drawn_image = cv.putText(drawn_image, "Passed: {}".format(results[0]), (80, 200), font, 0.3, (0, 255, 0), 1,
                             line)  # Prints the total number of passes for ORings
    drawn_image = cv.putText(drawn_image, "Failed: {}".format(results[1]), (80, 215), font, 0.3, (0, 0, 255), 1,
                             line)  # Prints the total number of fails for ORings

    # Optional for lines around the expected circle
    # radius = calculate_radius(image, centroid)
    # drawn_image = cv.circle(drawn_image, (centroid[1], centroid[0]), round(radius[0]), (125,125,125), 1)
    # drawn_image = cv.circle(drawn_image, (centroid[1], centroid[0]), round(radius[1]), (125,125,125), 1)

    cv.imshow(title, drawn_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return results


########################################################
#################### Process begins ####################
########################################################

# Call the method to read in all the images, stores them in a list
# images = read_images(directory)
# images = glob.glob(
#     r'G:\\Microsee\\camera_calibration\\CameraCalibration-master\\src\\images\\calibrated/2/*.png')
images = glob.glob(r'./images/calibrated/2/*.png')
# 已校正图像
# Image counter
imageCount = 1

# Morphological structure which is used for erosion and dilation
morph_struct = [
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]

# Results list which contains the total number of ORings which have passed and the time for processing
results = [0, 0, 0]

# Loops through the list of oring images scanned
for i in images:
    # image= cv.imread(i,0)
    # cv.imshow(i, image)
    # cv.waitKey(0)
    image = cv.imread(i)
    # Histogram plotting for the current image in the for loop
    plt.plot(image_hist(image))
    plt.show()

    start = time.time()
    image_threshold = threshold(image)
    cv.imshow('image_threshold', image_threshold)
    cv.waitKey(0)

    # image_threshold = closing(image_threshold, morph_struct)
    # cv.imshow('image_closing', image_threshold)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    image_labels = component_label(image_threshold)
    cv.imshow('component_label', image_labels)
    cv.waitKey(0)
    image_labels = remove_smallest_areas(image_labels)
    cv.imshow('remove_smallest_areas', image_labels)
    cv.waitKey(0)
    centroid = get_centroid(image_labels)
    bounding_box = make_bounding_box(image_labels)
    results = start_process(i.split('/')[-1], image_threshold, image_labels, centroid, bounding_box, start, results)

    # Print the processing statistics to the terminal
    print("---------- ORing Processing Results ----------")
    print("Time: {} seconds".format(results[2]))
    print("Passed ORings: {}".format(results[0]))
    print("Failed ORings: {}".format(results[1]))
    print("Images complete: {}".format(imageCount))
    imageCount += 1