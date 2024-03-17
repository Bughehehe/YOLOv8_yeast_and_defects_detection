import streamlit as st
from ultralytics import YOLO
import cv2
import os
import shutil
import numpy as np

model = YOLO("model_yeast.pt")

dir_path = os.path.dirname(os.path.abspath("web_api.ipynb"))
if(not os.path.exists(dir_path + "/cells_counted")): os.makedirs(dir_path + "/cells_counted")
if(os.path.exists(dir_path + "/runs")): shutil.rmtree(dir_path + "/runs", ignore_errors=True)


def main():
    st.title("ML model for yeast/infection detection and counting")
    st.subheader("Made by: Aleksander Kołodziejczyk and Rafał Ignacy")
    
    # Wczytaj obrazek
    st.header("")
    st.header("Inject your microscope photo")
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Wyświetl wczytany obrazek
        st.header("Original image")
        st.image(uploaded_image, caption="Original photo", use_column_width=True)
        
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        results = model.predict(image, save=True)
        result = results[0]

        count_yeast = 0
        count_infection = 0

        for item in result.boxes:
            if(item.cls[0].item() == 0):
                count_infection +=1
            else: count_yeast += 1


        image = "runs/detect/predict/image0.jpg"
        image_trans = cv2.imread(image, cv2.IMREAD_COLOR)


        box = image_trans[0:52, 0:265]
        box_parameters = box.shape
        for n in range(box_parameters[0]):
            for y in range(box_parameters[1]):
                box[n][y] = (0,0,0)

        image_trans = "runs/detect/predict/image0.jpg"
        image_trans = cv2.imread(image_trans)
        image_trans = cv2.putText(image_trans, "Counted Infested:   " + str(count_infection), (5,45), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        if(count_yeast < 10):
            image_trans = cv2.putText(image_trans, "Counted Yeast:     " + str(count_yeast), (5,20), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        elif(count_yeast >= 10 and count_yeast < 100):
            image_trans = cv2.putText(image_trans, "Counted Yeast:    " + str(count_yeast), (5,20), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        else: image_trans = cv2.putText(image_trans, "Counted Yeast:   " + str(count_yeast), (5,20), cv2.FONT_HERSHEY_SIMPLEX , 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        # Wyświetl zmodyfikowany obrazek
        image_trans = cv2.cvtColor(image_trans, cv2.COLOR_BGR2RGB)
        st.header("Processed image")
        st.image(image_trans, caption="Processed image", use_column_width=True)

        

if __name__ == "__main__":
    main()
