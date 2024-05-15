
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


json_file = open('C:\\Users\slhem\\Downloads\Emotion_detection_with_CNN(2)\Emotion_detection_with_CNN-main\model\\emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)


emotion_model.load_weights("C:\\Users\slhem\\Downloads\Emotion_detection_with_CNN(2)\Emotion_detection_with_CNN-main\model\\emotion_model.h5")
print("Loaded model from disk")


test_data_gen = ImageDataGenerator(rescale=1./255)


test_generator = test_data_gen.flow_from_directory(
        'C:\\Users\slhem\\Downloads\Emotion_detection_with_CNN(2)\Emotion_detection_with_CNN-main\data\\test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


predictions = emotion_model.predict_generator(test_generator)



print("-----------------------------------------------------------------")

c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print(c_matrix)
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=emotion_dict)
cm_display.plot(cmap=plt.cm.Blues)
plt.show()

# Classification report
print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predictions.argmax(axis=1)))




