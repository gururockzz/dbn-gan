from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
loaded_model = load_model('model')
img = image.load_img('results/S010_004_00000017_rlt.png', target_size=(48, 48))
emotion_mapping = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral', 7: 'Contempt'}
img = img.convert('L')
img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array /= 255.

predictions = loaded_model.predict(img_array)

predicted_index = np.argmax(predictions[0])

predicted_emotion = emotion_mapping[predicted_index]

print(f"Predicted emotion: {predicted_emotion}")



