import numpy as np
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('Carregando dados........')
y_train = pd.read_csv("train_labels.csv")
print('y train carregado')
x_train = pd.read_parquet("./train/", engine = 'fastparquet')
print('x train carregado')
print(x_train.head())
print(y_train.head())
x_train = x_train.merge(y_train, right_on='customer_ID', left_on='customer_ID')
print('merged!')
drop_columns = x_train.columns[:2]
print('dataset carregado!')
#split x_train in 3 new divisions, train, test, validation.
def get_dataset_partitions_pd(df, y_train, train_split=0.7, val_split=0.2, test_split=0.1):
    assert (train_split + test_split + val_split) == 1

    #specify seed to always have the same split distribution between runs
    customer_ids = y_train.sample(frac=1, random_state=7)['customer_ID'].values
    #splitting
    #train
    train_ds = df[df['customer_ID'].isin(customer_ids[:int(train_split * len(customer_ids))])]   
    
    #val
    val_ds = df[df['customer_ID'].isin(customer_ids[int(train_split * len(customer_ids)):
                                       int(train_split * len(customer_ids))+int(val_split*len(customer_ids))])]   
    #test
    test_ds = df[df['customer_ID'].isin(customer_ids[int(train_split * len(customer_ids))+int(val_split*len(customer_ids)):])]
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = get_dataset_partitions_pd(x_train, y_train)

train_ds = train_ds.drop(drop_columns ,axis=1)
val_ds = val_ds.drop(drop_columns ,axis=1)
test_ds = test_ds.drop(drop_columns ,axis=1)

del y_train
print('Dataset dividido')
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                   train_df=train_ds, val_df=val_ds, test_df=test_ds,
                   label_columns=['target']):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
          self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = 13

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    def split_window(self, features):
        inputs = features[:, self.input_slice, :-1]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(data=data,
                                                          targets=None,
                                                          sequence_length=self.total_window_size,
                                                          sequence_stride=1,
                                                          shuffle=False,
                                                          batch_size=32,)
        ds = ds.map(self.split_window)
            
        return ds
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
            
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)


w1 = WindowGenerator(input_width=13, label_width=1, shift=0)

MAX_EPOCHS = 5
def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=tf.keras.metrics.Recall(),
                                                      patience=patience,
                                                      mode='min')
    
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),   optimizer=tf.keras.optimizers.AdamW(use_ema=True),
                  metrics=['acc',tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
    
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,verbose=1,
                        callbacks=[early_stopping])
    return history


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.8, recurrent_dropout=0.8),
    tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.8, recurrent_dropout=0.8),
    #tf.keras.layers.Flatten(),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=128, activation= tf.keras.activations.relu),
    tf.keras.layers.Dense(units=64, activation= tf.keras.activations.relu),
    tf.keras.layers.Dense(units=1, activation= tf.keras.activations.sigmoid)
])
val_performance = {}
performance = {}
print('Começou o treinamento!')
history = compile_and_fit(lstm_model, w1)
val_performance['LSTM-16unitsSGD'] = lstm_model.evaluate(w1.val)
performance['LSTM-16unitsSGD'] = lstm_model.evaluate(w1.test, verbose=0)

lstm_model.save('LSTM')