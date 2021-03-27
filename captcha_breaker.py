from selenium import webdriver
from keras import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
import cv2
import pickle
import imutils
import numpy as np
import time
import urllib.request
import os


def load_classifier():
    num_classes = 33

    CNN_model = Sequential()
    CNN_model.add(
        Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 3), activation="relu"))
    CNN_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    CNN_model.add(
        Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 3), activation="relu"))
    CNN_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    CNN_model.add(Flatten())
    CNN_model.add(Dense(512, activation="relu"))
    CNN_model.add(Dense(num_classes, activation="softmax"))

    CNN_model.load_weights("weights.h5")
    binarizer = pickle.load(open("binarizer.pkl", "rb"))  
    return CNN_model, binarizer

#=====================================================
# Helper functions to find contours of the characters
#=====================================================
    
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def threshold(grayscaled):
    return cv2.threshold(
       grayscaled, 140, 255, cv2.THRESH_BINARY_INV
    )[1]
   

def dilate_characters(thresholded):
    kernel = np.ones((2,2), np.uint8)
    return cv2.dilate(thresholded, kernel, iterations = 1)

def find_contours(dilated):
    return cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]

#======================================================
# Helper functions to cut out the individual characters
#======================================================

def compute_bounding_rectangles(contours):
    return list(map(cv2.boundingRect, contours))


# If the detected rectangles are too close together, they will be
# combined into one
def merge_rectangles(rectangles):
    # Combines two rectangles into one that encompasses both
    def merge(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x = min(x1, x2)
        y = min(y1, y2)
        w = max(x1 + w1, x2 + w2) - x
        h = max(y1 + h1, y2 + h2) - y
        return (x, y, w, h)
    
    # Closeness according to x-axis
    def is_close(rect1, rect2):
        x1 = rect1[0]
        w1 = rect1[2]
        x2 = rect2[0]
        return abs(x1 - x2) <= 10 or abs(x1 + w1 - x2) <= 10
    
    # Rectangles should not be merged, if they are way above each other!
    # Closeness according to y-axis
    def way_above(rect1, rect2):
        x1 = rect1[0]
        x2 = rect2[0]
        y1 = rect1[1]
        y2 = rect2[1]
        w1 = rect1[2]
        h2 = rect2[3]
        return abs(y1 - y2) >= 20
        
    
    r = sorted(rectangles, key=lambda x: x[0]) # sort by x coordinate
    
    i = 0
    while (i < len(r) - 1):
        if is_close(r[i], r[i + 1]) and not way_above(r[1], r[i + 1]):
            r[i] = merge(r[i], r[i + 1])
            del r[i + 1]
        else:
            i += 1
            
    return r

# Cuts out the characters according to the bounding rectangles
def get_character_images(rectangles, image):
    char_images = []
    for rect in rectangles:
        x, y, w, h = rect
        char_image = image[y - 1 : y + h + 1, x - 1 : x + w + 1]
        char_images.append(char_image)
    return char_images

#======================================================
# The whole preprocessing pipeline
#======================================================

def preprocessing(img_file):
    # Determine contours
    img = cv2.imread(img_file)
    grayscaled = grayscale(img)
    thresholded = threshold(grayscaled)
    dilated = dilate_characters(thresholded)
    contours = find_contours(dilated)
    
    # Compute bounding rectangles
    rectangles = compute_bounding_rectangles(contours)
    rectangles = [rect for rect in rectangles if not (rect[2] <= 5 and rect[3] <=5)]
    rectangles = merge_rectangles(rectangles)
    
    # Cut out and save individual character images
    character_images = get_character_images(
        rectangles, dilated
    )

    return character_images


def normalize_dimensions(image, desired_width=20, desired_height=20):
    (h, w) = image.shape[:2]
    if w > h:
        image = imutils.resize(image, width=desired_width)
    else:
        image = imutils.resize(image, height=desired_height)
    width_padding = int((desired_width - image.shape[1]) / 2)
    height_padding = int((desired_height - image.shape[0]) / 2)
    WHITE = [255, 255, 255]
    image_with_border = cv2.copyMakeBorder(image, height_padding, height_padding, width_padding, width_padding, cv2.BORDER_CONSTANT, value=WHITE)
    image_with_border_resized = cv2.resize(image_with_border, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
    return image_with_border_resized



driver = webdriver.Chrome(executable_path="/Users/henryhelm/Desktop/captcha-bypasser/chromedriver")
driver.get("http://localhost:8000")

while True:
    # Download captcha image
    img_element = driver.find_element_by_id("captcha")
    src = img_element.get_attribute("src")
    urllib.request.urlretrieve(src, "captcha.png")

    # Extract character images
    character_images = preprocessing("captcha.png")

    if len(character_images) == 6: # has to be 6 characters or extraction was not successful
        # Save character images
        for index, char_image in enumerate(character_images):     
            image_save_path = os.path.join("", str(index) + ".png")
            try:
                cv2.imwrite(image_save_path, char_image)
            except:
                pass

        # Normalize dimensions of character images
        images = []
        for i in range(6):
            img = cv2.imread(str(i) + ".png")
            image_normalized = normalize_dimensions(img)
            images.append(image_normalized)

        # Use model to predict the captcha text
        model, binarizer = load_classifier()
        X = np.array(images, dtype="float") / 255.0
        predicted_text = "".join(binarizer.inverse_transform(model.predict(X)))

        # Write prediction into text field and submit
        input_field = driver.find_element_by_id("code")
        input_field.click()
        input_field.send_keys(predicted_text)
        driver.find_element_by_id("submit").click()

        submit_message = driver.find_element_by_id("submit-message").text
        if submit_message == "Bypassed CAPTCHA successfully :)":
            break # success!

     # Reload page to get a new captcha
    driver.find_element_by_id("reload").click()
        

time.sleep(10)
driver.close()    
