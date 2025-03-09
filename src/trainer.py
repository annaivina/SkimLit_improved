import tensorflow as tf
from sklearn.preprocessing import LabelEncoder 
import pandas as pd

class Trainer:

    def __init__(self, model, train_data, valid_data, test_data, epochs=0, callbacks=None):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.epochs = epochs
        self.callbacks = callbacks
        self.encoder_label = LabelEncoder()


    def train(self):

        print("Fitting the model ...")
        history = self.model.fit(
            self.train_data,
            epochs = self.epochs,
            validation_data = self.valid_data,
            callbacks=self.callbacks)
        
        return history
    

    def evaluate(self, model_name):

        print("Evaluating the model ...")
        results = self.model.evaluate(self.test_data)
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}")

        print("Predicting on the test data")

        y_pred_probs = self.model.predict(self.test_data)
        y_pred = tf.argmax(y_pred_probs, axis=1)

        train_label_encoded = self.encoder_label.fit_transform(self.train_data.map(lambda x: x['labels'])).to_numpy()
        test_label_encoded = self.encoder_label.transform(self.test_data.map(lambda x: x['labels'])).to_numpy()


        df = pd.DataFrame({"y_true": test_label_encoded, "y_pred": y_pred})
        print(f"Saving predictions to predictions_{model_name}.csv")
        df.to_csv("predictions"+model_name, index=False)
    

