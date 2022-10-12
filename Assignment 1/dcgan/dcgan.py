import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
# from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time


class DCGAN(keras.Model):
    def __init__(self, latent_dim=100):
        super(DCGAN, self).__init__()
        self.generator: keras.Model = self.make_generator()
        self.discriminator: keras.Model = self.make_discriminator()
        self._gen_opt = None
        self._disc_opt = None
        self._loss_fn = None
        self._gen_loss_metric = None
        self._disc_loss_metric = None
        self._latent_dim = latent_dim

    def make_generator(self):
        # override this
        pass

    def make_discriminator(self):
        # override this
        pass

    def compile(self, gen_opt, disc_opt, loss_fn, *args, **kwargs):
        super(DCGAN, self).compile(*args, **kwargs)
        self._gen_opt = gen_opt
        self._disc_opt = disc_opt
        self._loss_fn = loss_fn
        self._gen_loss_metric = tf.metrics.Mean(name='gen_loss')
        self._disc_loss_metric = tf.metrics.Mean(name='disc_loss')
    
    @tf.function
    def train_step(self, image_batch: tf.Tensor):
        batch_size = image_batch.shape[0]
        # print(image_batch)

        # ======== Train discriminator ================
        latent_vectors = tf.random.normal((batch_size, self._latent_dim))
        generated_images = self.generator(latent_vectors)

        with tf.GradientTape() as disc_tape:
            predictions = self.discriminator(tf.concat((generated_images, image_batch), 0), training=True)
            loss = self._loss_fn(
                tf.concat((tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))), 0),  # true labels
                predictions
            )

        gradients = disc_tape.gradient(loss, self.discriminator.trainable_weights)
        self._disc_opt.apply_gradients(zip(gradients, self.discriminator.trainable_weights))
        
        self._disc_loss_metric.update_state(loss)
        
        # ======= Train generator ======================
        latent_vectors = tf.random.normal((batch_size, self._latent_dim))
        
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator(latent_vectors, training=True)
            predictions = self.discriminator(generated_images)
            loss = self._loss_fn(
                tf.ones((batch_size, 1)),  # true labels
                predictions
            )
            
        gradients = gen_tape.gradient(loss, self.generator.trainable_weights)
        self._gen_grads = gradients
        self._gen_opt.apply_gradients(zip(gradients, self.generator.trainable_weights))
        
        self._gen_loss_metric.update_state(loss)
        
        return {
            'gen_loss': self._gen_loss_metric.result(),
            'disc_loss': self._disc_loss_metric.result()
        }
    
    def call(self, *args, **kwargs):
        pass


class SaveImage(keras.callbacks.Callback):
    def __init__(self, seed, loc='images', interval=10):
#         super(SaveImage, self).__init__()
        self.seed = seed
        self._image_save_loc = Path(loc)
        self._interval = interval
        if not os.path.exists(self._image_save_loc):
            os.mkdir(self._image_save_loc)
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self._interval != 0:
            return
        
        generated_images = self.model.generator(self.seed)
        
        n_rows = int(np.floor(np.sqrt(self.seed.shape[0])))
        n_cols = int(np.ceil(np.sqrt(self.seed.shape[0])))
        fig = plt.figure()
        
        for i in range(1, self.seed.shape[0]+1):
            plt.subplot(n_rows, n_cols, i)
            plt.imshow((generated_images[i-1] + 1) / 2)
            plt.axis(False)
            
        plt.show()
            
        fig.savefig(self._image_save_loc / f'image_at_epoch_{epoch}.png')
