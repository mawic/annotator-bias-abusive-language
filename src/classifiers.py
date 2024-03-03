import ktrain
from ktrain import text

import tensorflow as tf
import keras
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import uuid
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

import src.data.data_loader as dl

from definitions import MODEL_DIR, DATA_DIR


def compute_distance_matrix(ltnet):
    num_annotators = len(ltnet.annotators)
    bias_matrices = ltnet.model.get_weights()[-len(ltnet.annotators):]
    dist_mat = np.zeros((num_annotators, num_annotators))

    for i in tqdm(range(num_annotators)):
        for j in range(num_annotators):
            mat_i = bias_matrices[i]
            mat_j = bias_matrices[j]
            mat_diff = mat_i - mat_j
            mat_distance = np.linalg.norm(mat_diff)
            dist_mat[i,j] = mat_distance
    return dist_mat

def plot_distance_matrix(distance_matrix, ltnet):
    mask = np.zeros_like(distance_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False

    fig, ax = plt.subplots(figsize=(20,15))
    ax = sns.heatmap(distance_matrix, annot=True, ax=ax, mask=mask)
    ax.set_title(f"Distance between annotators' bias matrices", {'fontsize': 'large', 'fontweight': 'bold'})
    ax.set_xticklabels(ltnet.annotators, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ltnet.annotators, rotation=0, ha="right", rotation_mode="anchor")
    plt.show(ax)


def visualize_conf_matrix(davidson_cm, founta_cm, olid_cm, train_classifier, labels):
    f, (axD, axF, axO) = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12,3))
    sns.heatmap(davidson_cm, ax=axD, vmin=0, vmax=1, annot=True, cbar=False)
    sns.heatmap(founta_cm, ax=axF, vmin=0, vmax=1, annot=True, cbar=False)
    sns.heatmap(olid_cm, ax=axO, vmin=0, vmax=1, annot=True, cbar=False)

    axD.set_title(f'Train: {train_classifier}\nTest: Davidson', {'fontsize': 'large', 'fontweight': 'bold'})
    axF.set_title(f'Train: {train_classifier}\nTest: Founta', {'fontsize': 'large', 'fontweight': 'bold'})
    axO.set_title(f'Train: {train_classifier}\nTest: Olid', {'fontsize': 'large', 'fontweight': 'bold'})

    axD.set_xlabel('Predicted label')
    axD.set_ylabel('True label')

    axF.set_xlabel('Predicted label')
    axO.set_xlabel('Predicted label')

    axD.set_xticklabels(labels, rotation=0)
    axD.set_yticklabels(labels, rotation=90, ha="center", rotation_mode="anchor")
    plt.tight_layout()
    plt.show(f)

def evaluate_loaded_predictor(predictor, test_data):
    y_pred = predictor.predict(test_data["text"].values)
    y_true = test_data["label"].values
    report = classification_report(y_true, y_pred, target_names=predictor.preproc.get_classes(), digits=4)
    print(report)
    cm = confusion_matrix(y_true,  y_pred, normalize="true", labels=predictor.preproc.get_classes())
    print(cm)
    return cm

def apply_pseudolabeling(classifier_dict):
    """
    data_dict has to have a dataset as the key and the path to the classifier trained on the dataset as
    the value.

    """
    results = []
    for outer_dataset, outer_classifier_path in classifier_dict.items():
        # Load train set
        train_set = dl.load_preprocessed_data(outer_dataset, train=True, add_dataset_name = True)
        for inner_dataset, inner_classifier_path in classifier_dict.items():
            if inner_dataset == outer_dataset:
                continue
            else:
                classifier = ktrain.load_predictor(inner_classifier_path)
                print(f"Applying {inner_dataset} classifier on {outer_dataset} train set")
                # print(train_set)
                # print(train_set["text"].values)
                # train_set["text"].values
                pseudolabels = classifier.predict(train_set["text"].fillna('').values)
                # print(pseudolabels)
                train_set[f"{inner_dataset}_pseudolabels"] = pseudolabels
        results.append(train_set)
    return pd.concat(results).reset_index(drop=True)


class Classifier():

    def __init__(self, data, max_seq_len=265, max_features=5000, val_size=0.1, seed=2):
        self.data = data
        self.max_seq_len = max_seq_len # Length of the input sentences. Shorter texts will be padded longer will be cut
        self.max_features = max_features # This is the vocab size
        self.val_size = val_size
        self.seed = seed
        self.classes = data["label"].unique()
        self.preprocess() # Initialize train and validation set
        self.learner = None

    def preprocess(self):
        """
        Split the text in train and validation set and applying the
        preprocessing (tokenization etc.) as required for each classifier.
        """
        print(f"Tokenizing for {self.model_name} with maxlen = {self.max_seq_len} and classes = {self.classes}")
        trn, val, preproc = text.texts_from_df(self.data,
                                       text_column="text",
                                       label_columns=["label"],
                                       preprocess_mode=self.preprocess_mode,
                                       maxlen=self.max_seq_len,
                                       max_features=self.max_features,
                                       val_pct=self.val_size,
                                       random_state=self.seed)
        self.train_data = trn
        self.val_data = val
        self.preproc = preproc


    def show_model_summary(self):
        self.model.summary()

    def find_lr(self, max_epochs=2, batch_size = 64):
        if not self.learner:
            self.init_learner(batch_size=batch_size)
        self.learner.lr_find(show_plot=True, max_epochs=max_epochs)

    def init_learner(self, batch_size=64):
        self.learner = ktrain.get_learner(
            self.model,
            train_data=self.train_data,
            val_data=self.val_data,
            batch_size=batch_size)

        self.predictor = ktrain.get_predictor(self.learner.model, self.preproc)

    def fit(self, lr=3e-5, batch_size=64, epochs=2, early_stopping=4, reduce_on_plateau=2 ):

        if not self.learner:
            self.init_learner(batch_size=batch_size)

        #TODO: Use autofit with early stopping?
        if epochs:
            print(f"Start {self.model_name} training with lr = {lr}, batch_size = {batch_size} and epochs = {epochs}")
            self.learner.autofit(lr, epochs)
        else: # Use early stopping
            print(f"Start {self.model_name} training with lr = {lr}, batch_size = {batch_size} and early_stopping")
            self.learner.autofit(lr, early_stopping=early_stopping, reduce_on_plateau=reduce_on_plateau)

        self.predictor = ktrain.get_predictor(self.learner.model, self.preproc)
        # Validate on training on val_data
        #cm = self.learner.validate(class_names=self.preproc.get_classes())
        cm = self.learner.validate() # <-- To solve the problem with missing classes
        print(cm)
        # Show loss during training
        self.learner.plot()

    def save_predictor(self, experiment_name,  dataset_name, classifier_id=uuid.uuid4()):
        predictor_name = f"{self.model_name}-{dataset_name}-{classifier_id}"
        predictor_path = os.path.join(MODEL_DIR, experiment_name, predictor_name)
        self.predictor.save(predictor_path)
        print(f"Saved model at {predictor_path}")

    def predict(self, data):
        print(data)
        return self.predictor.predict(data["text"].values)

    def get_classes(self):
        return self.preproc.get_classes()

    def evaluate(self, data):
        y_pred = self.predict(data)
        y_true = data["label"].values

        report = classification_report(y_true, y_pred, target_names=self.classes, digits=4)
        print(report)
        cm = confusion_matrix(y_true,  y_pred,  normalize="true", labels=self.classes)
        print(cm)
        return cm


class DistilBertClassifier(Classifier):
    def __init__(self, data, max_seq_len=265, max_features=5000, val_size=0.1, seed=2):
        self.model_name = "distilbert"
        self.preprocess_mode = "distilbert"

        super().__init__(
            data=data,
            max_seq_len=max_seq_len,
            max_features=max_features,
            val_size=val_size,
            seed=seed
        )

        self.model = text.text_classifier('distilbert',
                                            self.train_data,
                                            preproc=self.preproc)

class BertClassifier(Classifier):
    def __init__(self, data, max_seq_len=265, max_features=5000, val_size=0.1, seed=2):
        self.model_name = "bert"
        self.preprocess_mode = "bert"

        super().__init__(
            data=data,
            max_seq_len=max_seq_len,
            max_features=max_features,
            val_size=val_size,
            seed=seed
        )

        self.model = text.text_classifier('bert', self.train_data, preproc=self.preproc)

class BiLSTMClassifier(Classifier):
    def __init__(self, data, max_seq_len=265, max_features=5000, val_size=0.1, seed=2):
        self.model_name = "BiLSTM"
        self.preprocess_mode = "standard" # Use standard tf.Tokenizer

        super().__init__(
            data=data,
            max_seq_len=max_seq_len,
            max_features=max_features,
            val_size=val_size,
            seed=seed
        )

        self.model = self._load_model()

    def _load_embeddings(self, embedding_dim):
        print("Loading word vectors")
        path_to_glove_file = os.path.join(DATA_DIR, "embeddings", f"glove.6B.{embedding_dim}d.txt")

        embeddings_index = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        print("Loaded %s word vectors." % len(embeddings_index))

        word_index = self.preproc.tok.word_index
        num_tokens = len(word_index) + 1
        hits = 0
        misses = 0

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))
        return embedding_matrix, num_tokens

    def _load_model(self):
        embedding_dim = 50
        embedding_matrix, num_tokens = self._load_embeddings(embedding_dim)

        lstm_layer_nodes = 64
        dense1_layer_nodes = 16
        num_classes = len(self.classes)

        model = keras.Sequential()
        # without pre trained embeddings
        #model.add(keras.layers.Embedding(self.max_features, embedding_dim, input_length=self.max_seq_len, mask_zero=True))
        model.add(keras.layers.Embedding(num_tokens,
                                         embedding_dim,
                                         embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                         trainable=False))

        model.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_layer_nodes)))
        model.add(keras.layers.Dense(dense1_layer_nodes, activation=tf.nn.relu))
        model.add(keras.layers.Dense(num_classes, activation=tf.nn.softmax))
      #  sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['categorical_accuracy'])
        return model

class ProbabilityTransitionLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(ProbabilityTransitionLayer, self).__init__()
        self.num_outputs = num_outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({"num_outputs": self.num_outputs})
        return config

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),self.num_outputs],
                                     initializer="glorot_normal",
                                      trainable=True)

        diagonal = np.asarray([999] * self.num_outputs)
        new_kernel = tf.linalg.set_diag(self.kernel, diagonal)
        K.update(self.kernel, new_kernel)

        self._normalize_kernel()


    # @tf.function
    def _normalize_kernel(self):
        kernel_clipped = K.clip(self.kernel, 0.00001, 1000)
        tensor_normalized, norm = tf.linalg.normalize(kernel_clipped, ord=1, axis=-1)
        K.update(self.kernel, tensor_normalized)

    def call(self, input):
        self._normalize_kernel()
        output = K.dot(input, self.kernel)
        return output

def masked_sparse_categorical_crossentropy(y_true, y_pred, mask_value=-1):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    ytrue_masked = y_true * mask
    ypred_masked = y_pred * mask
    return K.sparse_categorical_crossentropy(ytrue_masked,  ypred_masked)

def masked_sparse_categorical_accuracy(y_true, y_pred, mask_value=-1):
    mask = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    ytrue_masked = y_true * mask
    ypred_masked = y_pred * mask
    return keras.metrics.sparse_categorical_accuracy(ytrue_masked,  ypred_masked)

class LTNetClassifier():
    def __init__(self, base_model_path, data, val_size=0.1, seed=2):
        self.base_model_path = base_model_path
        self.data = data
        self.seed = seed
        self.val_size = val_size

        # Load base model
        self.base_classifier = ktrain.load_predictor(self.base_model_path)
        self.base_model = self.base_classifier.model
        self.base_preproc = self.base_classifier.preproc
        self.classes = self.base_preproc.get_classes()
        self.annotators = self.data.iloc[:, 1:].columns

        self._preprocess_data()
        self.model = self._init_bias_matrix()

        print(np.array(self.train_text).shape, np.array(self.train_labels).T.shape)
        self.learner = ktrain.get_learner(self.model, train_data=(np.array(self.train_text), np.array(self.train_labels).T), val_data=(np.array(self.val_text),np.array(self.val_labels).T))

    def _preprocess_data(self):
        """
        Data expected in the dataframe from "text", "annotator1, annotator2, ..."
        """
        self.data = self.data.replace({np.nan: "zzz"})
        X_train, X_val, y_train, y_val = train_test_split(self.data["text"].values, self.data.iloc[:, 1:], test_size=self.val_size, random_state=self.seed)
        if "distil" in self.base_model.name:
            self.train_text = self.base_preproc.preprocess_test(X_train).x
            self.train_input_ids = self.train_text[:, 0, :]
            self.train_attention_mask = self.train_text[:, 1, :]

            self.val_text = self.base_preproc.preprocess_test(X_val).x
            self.val_input_ids = self.val_text[:, 0, :]
            self.val_attention_mask = self.val_text[:, 1, :]

        else:
            self.train_text = self.base_preproc.preprocess_test(X_train)[0]
            self.val_text = self.base_preproc.preprocess_test(X_val)[0]

        print(self.base_preproc.ytransform.c)
        le = LabelEncoder().fit([*self.base_preproc.ytransform.c, "zzz"])
        y_train = y_train.apply(lambda x: le.transform(x))
        y_val = y_val.apply(lambda x: le.transform(x))

        largest_index = len(le.classes_)-1
        y_train = y_train.replace({largest_index: -1})
        y_val = y_val.replace({largest_index:-1})

        self.train_labels = list(y_train.values.T.astype("float32"))
        self.val_labels = list(y_val.values.T.astype("float32"))

    def _init_bias_matrix(self):
        if "distil" in self.base_model.name:
            input_ids = tf.keras.Input(shape=(self.base_preproc.maxlen, ),dtype='int32')
            attention_mask = tf.keras.Input(shape=(self.base_preproc.maxlen, ), dtype='int32')
            output = self.base_classifier.model([input_ids, attention_mask])[0]
            base_model = tf.keras.Model(inputs=[input_ids,attention_mask], outputs=output)
        else:
            base_model = self.base_model

        if "distil" in self.base_model.name:
            softmax_lt = keras.layers.Dense(len(self.classes), activation=tf.nn.softmax)(base_model.output)
            out = []
            for i in range(len(self.annotators)): # Num annotators
                # y = keras.layers.Dense(len(self.classes), activation=tf.nn.softmax)(base_model.output)
                y = ProbabilityTransitionLayer(len(self.classes))(softmax_lt)
                out.append(y)
        else:
            out = []
            for i in range(len(self.annotators)): # Num annotators
                # y = keras.layers.Dense(len(self.classes), activation=tf.nn.softmax)(base_model.output)
                y = ProbabilityTransitionLayer(len(self.classes))(base_model.output)
                out.append(y)

        if "distil" in self.base_model.name:
            model = tf.keras.Model(inputs=[input_ids,attention_mask], outputs=out)
        else:
            model = tf.keras.Model(self.base_model.input, outputs=out)

        model.compile(loss=masked_sparse_categorical_crossentropy, optimizer="sgd", metrics=[masked_sparse_categorical_accuracy])
        return model

    def find_lr(self, max_epochs=2, batch_size = 64):
        self.learner.lr_find(show_plot=True, max_epochs=max_epochs)

    def fit(self, lr=3e-5, batch_size=64, epochs=2, early_stopping=4, reduce_on_plateau=2 ):
        if epochs:
            #print(f"Start {self.model_name} training with lr = {lr}, batch_size = {batch_size} and epochs = {epochs}")
            if "distil" in self.base_model.name:
                hist = self.model.fit([self.train_input_ids, self.train_attention_mask], self.train_labels, batch_size=batch_size, epochs=epochs, validation_data=([self.val_input_ids, self.val_attention_mask], self.val_labels))
            else:
                hist = self.model.fit(self.train_text, self.train_labels, batch_size=batch_size, epochs=epochs, validation_data=(self.val_text, self.val_labels))

        else: # Use early stopping
           # print(f"Start {self.model_name} training with lr = {lr}, batch_size = {batch_size} and early_stopping")
            mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=early_stopping)
            if "distil" in self.base_model.name:
                hist = self.model.fit([self.train_input_ids, self.train_attention_mask], self.train_labels, batch_size=batch_size, epochs=1024, validation_data=([self.val_input_ids, self.val_attention_mask], self.val_labels), callbacks=[es, mc])
            else:
                hist = self.model.fit(self.train_text, self.train_labels, batch_size=batch_size, epochs=1024, validation_data=(self.val_text, self.val_labels), callbacks=[es, mc])

                 # self.model = load_model('best_model.h5')
        # plot training history
        plt.plot(hist.history['loss'], label='train')
        plt.plot(hist.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        # model.fit(train_padded, y_train_list,  validation_split=0.1, shuffle=True, verbose=2)
        #  hist = self.model.fit(self.train_text, self.train_labels, batch_size=batch_size, epochs=epochs, validation_data=(self.val_text, self.val_labels))

    def show_model_summary(self):
        self.model.summary()

    def predict_annotators(self, data):
        preproc_data = self.base_preproc.preprocess_test(data)[0]
        pred = self.model.predict(preproc_data)
        pred_class = [np.argmax(pre,axis=1) for pre in pred]
        return np.array(pred_class)

    def predict_truth(self, data, return_labels=False):
        preproc_data = self.base_preproc.preprocess_test(data)[0]

        if "distil" in self.base_model.name:
            text = self.base_preproc.preprocess_test(data).x
            input_ids = text[:, 0, :]
            attention_mask = text[:, 1, :]

        position_truth_layer = len(self.model.layers) - len(self.annotators) - 1


        base_model =  tf.keras.Model(inputs=self.model.input,
                                 outputs=self.model.layers[position_truth_layer].output)

        if "distil" in self.base_model.name:
            pred = base_model.predict([input_ids, attention_mask])
        else:
            pred = base_model.predict(preproc_data)

        pred_class = np.argmax(pred,axis=1)
        if return_labels:
           pred_class = [self.classes[x] for x in pred_class]
        return pred_class

    def evaluate(self, test_data):
        #y_true = test_data["label"]
        #print(y_true)
        le = LabelEncoder().fit(self.base_preproc.ytransform.c)
        y_true = le.transform(test_data["label"].values)
        #test_data["label"] = test_data["label"].apply(lambda x: print(x), raw=True)
        #y_true = test_data["label"].values
        X_test = test_data["text"].values
        y_pred = self.predict_truth(X_test)

        report = classification_report(y_true, y_pred, target_names=self.classes, digits=4)
        print(report)
        cm = confusion_matrix(y_true,  y_pred,  normalize="true")
        print(cm)
        return cm


    def _plot_bias_matrices(self, error_rates, observers, classes):

        for i, obs in enumerate(observers):
            error_rate = error_rates[i]

            ax = sns.heatmap(error_rate, vmin=0, vmax=1, annot=True, cbar=False)
            ax.set_title(f'{obs}', {'fontsize': 'large', 'fontweight': 'bold'})
            ax.set_xlabel('Observed label')
            ax.set_ylabel('Latent truth')
            ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticklabels(classes, rotation=0, ha="right", rotation_mode="anchor")
            plt.show(ax)

    def show_bias_matrices(self):
        self._plot_bias_matrices(
        error_rates = self.model.get_weights()[-len(self.annotators):],
        observers = self.annotators,
        classes = self.classes
        )
        # names = [weight.name for layer in self.model.layers for weight in layer.weights]
        # weights = self.model.get_weights()
        # names, weights = names[-len(self.annotators):], weights[-len(self.annotators):]
        # for id, (name, weight) in enumerate(zip(names, weights)):
        #     print(self.annotators[id], name, weight.shape)
        #     display(weight)
