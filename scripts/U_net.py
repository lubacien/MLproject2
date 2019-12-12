from tensorflow.python.keras import layers
from tensorflow.python.keras import models

img_shape = (400, 400, 3)

def make_model1():
    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder


    def encoder_block(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

        return encoder_pool, encoder

    def encoder_block_5x5(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((5, 5), strides=(5, 5))(encoder)

        return encoder_pool, encoder


    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def decoder_block_5x5(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (5, 5), strides=(5, 5), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder


    inputs = layers.Input(shape=img_shape)
    # 400
    print(inputs.shape)
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 200
    print(encoder0_pool.shape)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 100
    print(encoder1_pool.shape)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 50
    print(encoder2_pool.shape)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 25
    print(encoder3_pool.shape)
    encoder4_pool, encoder4 = encoder_block_5x5(encoder3_pool, 512)
    # 5
    print(encoder4_pool.shape)
    center = conv_block(encoder4_pool, 1024)
    # center
    print(center.shape)
    decoder4 = decoder_block_5x5(center, encoder4, 512)
    # 25
    print(decoder4.shape)
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    print(decoder3.shape)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    print(decoder2.shape)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    print(decoder1.shape)
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    print(decoder0.shape)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    print(outputs.shape)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def make_model2():
    def conv_block(input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder


    def encoder_block(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

        return encoder_pool, encoder

    def encoder_block_5x5(input_tensor, num_filters):
        encoder = conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((5, 5), strides=(5, 5))(encoder)

        return encoder_pool, encoder


    def decoder_block(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def decoder_block_5x5(input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (5, 5), strides=(5, 5), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    inputs = layers.Input(shape=img_shape)
    # 400
    print(inputs.shape)
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    # 200
    print(encoder0_pool.shape)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    # 100
    print(encoder1_pool.shape)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    # 50
    print(encoder2_pool.shape)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    # 25
    print(encoder3_pool.shape)
    encoder4_pool, encoder4 = encoder_block_5x5(encoder3_pool, 512)
    # 5
    print(encoder4_pool.shape)
    center = conv_block(encoder4_pool, 1024)
    # center
    print(center.shape)
    decoder4 = decoder_block_5x5(center, encoder4, 512)
    # 25
    print(decoder4.shape)
    decoder3 = decoder_block(decoder4, encoder3, 256)
    # 32
    print(decoder3.shape)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    # 64
    print(decoder2.shape)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    # 128
    print(decoder1.shape)
    decoder0 = decoder_block(decoder1, encoder0, 32)
    # 256
    print(decoder0.shape)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
    print(outputs.shape)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model