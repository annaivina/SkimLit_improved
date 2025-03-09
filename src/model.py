import tensorflow as tf # type: ignore
import tensorflow_hub as hub # type: ignore
import tensorflow_text as text # type: ignore
from tensorflow.keras.layers import Embedding, Dense, Dropout, Bidirectional, LSTM, Concatenate, Input, Lambda
import os


class SkimLitModel(tf.keras.Model):


    def __init__(self, lr, classes, dropout, bert_train=False):
        super(SkimLitModel, self).__init__()

        self.lr = lr
        self.ber_train = bert_train
        self.classes = classes
        self.dropout = dropout 


        #Load the models form the tf_hub
        self.bert_processes = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.bert_model_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3", 
                                               trainable=bert_train, 
                                               name="BERT")

        #Load the char embedding 
        self.char_embedding = Embedding(input_dim=98,#the number of token in vectorizer + 2 
                                        output_dim=25,# this is something taken form the PubMed paper 
                                        mask_zero=True,
                                        name='char_embedding'
                                    )
        

        self.dense_out = Dense(self.classes, activation='softmax', name="dense_output_layer")
        self.dropout = Dropout(dropout)
        self.word_model = self._create_word_model()
        self.char_model=self._create_char_model()
        self.line_model=self._create_line_model()
        self.total_model=self._create_total_model()
        self._build_model()


    
    def _create_word_model(self):
        word_input = Input(shape=(), dtype=tf.string, name='sentences')
        process_word = Lambda(lambda x: self.bert_processes(x), output_shape={"input_word_ids": (128,), "input_mask": (128,), "input_type_ids": (128,)})(word_input)
        bert_output = Lambda(lambda x: self.bert_model_layer(x), output_shape={"pooled_output": (768,), "sequence_output": (128, 768)})(process_word)
        word_output = Dense(64, activation='relu')(bert_output["pooled_output"])
        return tf.keras.Model(word_input, word_output, name='word model')
    
    def _create_char_model(self):
        ## Create char model
        char_input = Input(shape=(96,), dtype=tf.int64, name='chars')
        char_embed = self.char_embedding(char_input)
        char_bi_slstm = Bidirectional(LSTM(25))(char_embed)
        return tf.keras.Model(char_input, char_bi_slstm, name='char model')
    

    def _create_line_model(self):
        line_input = Input(shape=(15,), dtype=tf.float32, name='line_nb')#average number of lines 15
        line_output = Dense(32, activation='relu')(line_input)
        return tf.keras.Model(line_input, line_output, name="line model")
    
    def _create_total_model(self):
        #Create total lines mode
        total_input = Input(shape=(21,), dtype=tf.float32, name='total_lines')
        total_output = Dense(32, activation='relu')(total_input)
        return tf.keras.Model(total_input, total_output, name='total_line model')
    
    def _build_model(self):
        concat_1 = Concatenate(name="concat_layer_1")([self.word_model.output, self.char_model.output])

        #Make additional dense layer between concatinations
        z = Dense(64, activation='relu')(concat_1)
        #Use dropout before and after 2 concat (dorpoout of 0.5 and tweo layers turns out the best solution for overfitting for BERT)
        z = self.dropout(z)

        concat_2 = Concatenate(name='concat_layer_2')([self.line_model.output, self.total_model.output, z])
        #concat_2 = self.dropout(concat_2)
        output_layer = self.dense_out(concat_2)

        super().__init__(inputs={"line_nb": self.line_model.input, "total_lines": self.total_model.input, "sentences": self.word_model.input, "chars": self.char_model.input}, outputs=output_layer)
    

    def compile_model(self):

        self.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                     metrics = ['accuracy'])
        







        



