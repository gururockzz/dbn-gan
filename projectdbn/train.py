import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import sys

# Specify the file path (change '/content/fer2013.csv' to the actual file path)
file_path = 'ck+.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame to verify that the data has been loaded successfully
print(df.head())

# Extract images and labels
images = []
for image_pixels in df['pixels']:
    pixel_values = np.fromstring(image_pixels, dtype=int, sep=' ')

    # Check if the length is as expected (48*48)
    if len(pixel_values) == 48 * 48:
        image = pixel_values.reshape((48, 48, 1))
        images.append(image)

# Convert the list of images to a numpy array
images = np.array(images)
labels = to_categorical(df['emotion'], num_classes=8)  # Assuming 7 classes for emotions

# Normalize pixel values to the range [0, 1]
images = images / 255.0
# Split dataset into training and validation sets (Placeholder, replace with your actual splitting logic)
def split_dataset(images, labels, validation_split=0.2):
    num_samples = len(images)
    num_validation_samples = int(validation_split * num_samples)

    indices = np.random.permutation(num_samples)

    training_indices = indices[num_validation_samples:]
    validation_indices = indices[:num_validation_samples]

    training_images, training_labels = images[training_indices], labels[training_indices]
    validation_images, validation_labels = images[validation_indices], labels[validation_indices]

    return training_images, training_labels, validation_images, validation_labels

# Call the function to split the dataset
training_images, training_labels, validation_images, validation_labels = split_dataset(images, labels)

# Define a Deep Belief Network (DBN) class
class DeepBeliefNetwork(Model):
    def __init__(self, input_dim, num_classes):
        super(DeepBeliefNetwork, self).__init__()
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.output_layer = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

# Define RL agent class
class RLAgent(tf.keras.Model):
    def __init__(self, input_dim, output_dim):
        super(RLAgent, self).__init__()
        self.dbn_model = DeepBeliefNetwork(input_dim, num_classes=output_dim)
        self.output_layer = Dense(output_dim, activation='softmax')

    def call(self, inputs):
        dbn_output = self.dbn_model(inputs)
        return self.output_layer(dbn_output)

# Use the legacy optimizer for both models
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Initialize RL agent
num_actions = 8  # Adjust based on your RL problem and the actual number of classes in your dataset
num_emotions = 8  # Assuming the input dimension is the number of emotions
rl_agent = RLAgent(input_dim=num_emotions, output_dim=num_actions)

# Training loop...
num_epochs = 5
batch_size = 32

# Create training dataset using tf.data.Dataset
training_labels_indices = np.argmax(training_labels, axis=1)
training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_labels_indices)).shuffle(buffer_size=10000).batch(batch_size)



# Training loop...
for epoch in range(num_epochs):
    # Iterate over batches in the training set
    for batch in training_dataset:
        images, labels_indices = batch

        # Train RL agent
        with tf.GradientTape() as tape:
            rl_predictions = rl_agent(images)
            rl_loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels_indices, rl_predictions)
        gradients = tape.gradient(rl_loss_value, rl_agent.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rl_agent.trainable_variables))

        # Print predictions during training
        print(f"Predictions during training: {rl_predictions.numpy()}", flush=True)
        sys.stdout.flush()

    # Optionally, evaluate the DBN on the validation set and display detected emotions
    validation_predictions = rl_agent.dbn_model(validation_images)
    detected_emotion_indices = np.argmax(validation_predictions, axis=1)
    detected_emotion_names = detected_emotion_indices.tolist()  # Convert to a list of integers

    emotion_mapping = {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Sadness', 5: 'Surprise', 6: 'Neutral', 7: 'Contempt'}
    detected_emotion_names = [emotion_mapping[idx] for idx in detected_emotion_indices]

    print(f"Epoch {epoch+1}: RL Loss: {rl_loss_value}", flush=True)
    print(f"Detected Emotion Indices: {detected_emotion_indices}", flush=True)
    print(f"Detected Emotion Names: {detected_emotion_names}", flush=True)
    sys.stdout.flush()
    # Assuming 'rl_agent' is your model
rl_agent.save('model')



