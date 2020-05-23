# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:14:17 2020

@author: angel
"""

import tensorflow as tf

import numpy as np
import os
import time
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Leemos el archivo y despues se decodifica para py2 compat
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#  Length of text es la cantidad de caracteres que contiene
print ('Length of text: {} characters'.format(len(text)))

# Echa un vistazo a los primeros 250 caracteres en el texto
print(text[:250])
# Los caracteres únicos en el archivo
vocab = sorted(set(text))
print ('{} unique characters'.format(len(vocab)))
# Crear un mapeo de caracteres únicos a índices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')
# Muestra cómo los primeros 13 caracteres del texto se asignan a enteros
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# La oración de longitud máxima que queremos para una sola entrada en caracteres
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# crear ejemplos de entrenamineto/objetivos
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])
  
  sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(idx2char[item.numpy()])))
  
  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))
  
  for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    
    # Batch size
BATCH_SIZE = 64

# Tamaño del búfer para barajar el dataset
# (Los datos TF están diseñados para funcionar con secuencias posiblemente infinitas,
# para que no intente barajar toda la secuencia en la memoria. En cambio,mantiene un búfer en el que baraja elementos).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset

# Longitud del vocabulario en caracteres
vocab_size = len(vocab)

# La dimensión de incrustación
embedding_dim = 256

# Número de unidades RNN
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
  
  model.summary()
  
  sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())
model.compile(optimizer='adam', loss=loss)
# directorio donde se guardaran los checkpints
checkpoint_dir = './training_checkpoints'
# Nombre de los archivos chekpoint
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))
model.summary()
def generate_text(model, start_string):
 # Paso de evaluación (generar texto usando el modelo aprendido)

  # Número de caracteres a generar
  num_generate = 1000

  # Convertir nuestra cadena de inicio a números (vectorización)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Cadena vacía para almacenar nuestros resultados.
  text_generated = []

   # Las bajas temperaturas dan como resultado un texto más predecible.
   # Temperaturas más altas resultan en texto más sorprendente.
   # Experimente para encontrar la mejor configuración.
  temperature = 1.0

  # batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # eliminar la dimensión del batch
      predictions = tf.squeeze(predictions, 0)

     # usamos una distribución categórica para predecir el carácter devuelto por el modelo
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # Pasamos el carácter predicho como la siguiente entrada al modelo
      # junto con el estado oculto anterior
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"ROMEO: "))
model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(inp, target):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  return loss
# Pasos de entrenamiendo
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

 # inicializando el estado oculto al comienzo de cada epoch
   # Inicialmente oculto es Ninguno
  hidden = model.reset_states()

  for (batch_n, (inp, target)) in enumerate(dataset):
    loss = train_step(inp, target)

    if batch_n % 100 == 0:
      template = 'Epoch {} Batch {} Loss {}'
      print(template.format(epoch+1, batch_n, loss))

  # saving (checkpoint) the model every 5 epochs
  # guardardamos (chekpoint) el modelo cada 5 epochs
  if (epoch + 1) % 5 == 0:
    model.save_weights(checkpoint_prefix.format(epoch=epoch))

  print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
  print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))

    