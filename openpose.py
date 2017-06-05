import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.backend import set_image_data_format
set_image_data_format("channels_last")
from keras.models import Model
from keras.layers import Input
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Dropout, Reshape
from keras.layers.merge import Concatenate, Multiply
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D
from keras.backend import floatx
from keras.optimizers import SGD, Adam
import keras.backend as K


def stage1(input_tensor):
    x = input_tensor
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same",  activation="relu", name="conv5_1_CPM_L1")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same",  activation="relu", name="conv5_2_CPM_L1")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same",  activation="relu", name="conv5_3_CPM_L1")(x)
    x = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv5_4_CPM_L1")(x)
    x = Conv2D(38,  kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv5_5_CPM_L1")(x)
    affinity_fields = x # affinity fields

    x = input_tensor
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same",  activation="relu", name="conv5_1_CPM_L2")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same",  activation="relu", name="conv5_2_CPM_L2")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same",  activation="relu", name="conv5_3_CPM_L2")(x)
    x = Conv2D(512, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv5_4_CPM_L2")(x)
    x = Conv2D(19,  kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="conv5_5_CPM_L2")(x)
    parts = x # parts

    return [affinity_fields, parts] # L1, L2 (Loss1, Loss2)

def stageN(n: int, input_tensor):
    x = input_tensor
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv1_stage"+str(n)+"_L1")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv2_stage"+str(n)+"_L1")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv3_stage"+str(n)+"_L1")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv4_stage"+str(n)+"_L1")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv5_stage"+str(n)+"_L1")(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="Mconv6_stage"+str(n)+"_L1")(x)
    x = Conv2D(38,  kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="Mconv7_stage"+str(n)+"_L1")(x)
    affinity_fields = x # affinity fields

    x = input_tensor
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv1_stage"+str(n)+"_L2")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv2_stage"+str(n)+"_L2")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv3_stage"+str(n)+"_L2")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv4_stage"+str(n)+"_L2")(x)
    x = Conv2D(128, kernel_size=(7, 7), strides=(1, 1), padding="same",  activation="relu", name="Mconv5_stage"+str(n)+"_L2")(x)
    x = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="Mconv6_stage"+str(n)+"_L2")(x)
    x = Conv2D(19,  kernel_size=(1, 1), strides=(1, 1), padding="valid", activation="relu", name="Mconv7_stage"+str(n)+"_L2")(x)
    parts = x # parts

    return [affinity_fields, parts] # L1, L2 (Loss1, Loss2)

def create_openpose(train: bool, lr=0.00001) -> Model:
    input_tensor = Input(shape=(368, 368, 3))
    if train:
        vec_weight   = Input(shape=(46, 46, 38), name="vec_weight")
        heat_weight  = Input(shape=(46, 46, 19), name="heat_weight")

    # ideal input
    # [vec_weight, heat_weight, vec_temp, heat_temp] = Slice(name="vec_weight")(label)
    # label_vec  = Multiply(name="label_vec")([vec_weight, vec_temp]) # Eltwise PROD
    # label_heat = Multiply(name="label_heat")([heat_weight, heat_temp]) # Eltwise PROD

    x = input_tensor
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv1_1")(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv1_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool1_stage1")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv2_1")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv2_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool2_stage1")(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv3_1")(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv3_2")(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv3_3")(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv3_4")(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name="pool3_stage1")(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv4_1")(x)
    x = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv4_2")(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv4_3_CPM")(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", name="conv4_4_CPM")(x)
    relu4_4_CPM = x

    x = relu4_4_CPM # stage1
    [fields, parts] = stage1(x)
    [conv5_5_CPM_L1, conv5_5_CPM_L2] = [fields, parts]

    x = Concatenate(name="concat_stage2")([relu4_4_CPM, fields, parts])  # stage2
    [fields, parts] = stageN(2, x)
    [Mconv7_stage2_L1, Mconv7_stage2_L2] = [fields, parts]

    x = Concatenate(name="concat_stage3")([relu4_4_CPM, fields, parts])  # stage3
    [fields, parts] = stageN(3, x)
    [Mconv7_stage3_L1, Mconv7_stage3_L2] = [fields, parts]

    x = Concatenate(name="concat_stage4")([relu4_4_CPM, fields, parts])  # stage4
    [fields, parts] = stageN(4, x)
    [Mconv7_stage4_L1, Mconv7_stage4_L2] = [fields, parts]

    x = Concatenate(name="concat_stage5")([relu4_4_CPM, fields, parts])  # stage5
    [fields, parts] = stageN(5, x)
    [Mconv7_stage5_L1, Mconv7_stage5_L2] = [fields, parts]

    x = Concatenate(name="concat_stage6")([relu4_4_CPM, fields, parts])  # stage6
    [fields, parts] = stageN(6, x)
    [Mconv7_stage6_L1, Mconv7_stage6_L2] = [fields, parts]
    
    if train:
        weight_stage1_L1 = Multiply(name="weight_stage1_L1")([conv5_5_CPM_L1,    vec_weight])
        weight_stage1_L2 = Multiply(name="weight_stage1_L2")([conv5_5_CPM_L2,   heat_weight])
        weight_stage2_L1 = Multiply(name="weight_stage2_L1")([Mconv7_stage2_L1,  vec_weight])
        weight_stage2_L2 = Multiply(name="weight_stage2_L2")([Mconv7_stage2_L2, heat_weight])
        weight_stage3_L1 = Multiply(name="weight_stage3_L1")([Mconv7_stage3_L1,  vec_weight])
        weight_stage3_L2 = Multiply(name="weight_stage3_L2")([Mconv7_stage3_L2, heat_weight])
        weight_stage4_L1 = Multiply(name="weight_stage4_L1")([Mconv7_stage4_L1,  vec_weight])
        weight_stage4_L2 = Multiply(name="weight_stage4_L2")([Mconv7_stage4_L2, heat_weight])
        weight_stage5_L1 = Multiply(name="weight_stage5_L1")([Mconv7_stage5_L1,  vec_weight])
        weight_stage5_L2 = Multiply(name="weight_stage5_L2")([Mconv7_stage5_L2, heat_weight])
        weight_stage6_L1 = Multiply(name="weight_stage6_L1")([Mconv7_stage6_L1,  vec_weight])
        weight_stage6_L2 = Multiply(name="weight_stage6_L2")([Mconv7_stage6_L2, heat_weight])

        model = Model(inputs=[input_tensor, vec_weight, heat_weight], outputs=[
            weight_stage1_L1,
            weight_stage1_L2,
            weight_stage2_L1,
            weight_stage2_L2,
            weight_stage3_L1,
            weight_stage3_L2,
            weight_stage4_L1,
            weight_stage4_L2,
            weight_stage5_L1,
            weight_stage5_L2,
            weight_stage6_L1,
            weight_stage6_L2,
        ])
        # optimizer = Adam(lr=lr)
        model.compile(
            optimizer=SGD(lr=lr, momentum=0.9, decay=0.0005, nesterov=True),
            loss={
                'weight_stage1_L1': 'mean_squared_error', # label_vec
                'weight_stage1_L2': 'mean_squared_error', # label_heat
                'weight_stage2_L1': 'mean_squared_error', # label_vec
                'weight_stage2_L2': 'mean_squared_error', # label_heat
                'weight_stage3_L1': 'mean_squared_error', # label_vec
                'weight_stage3_L2': 'mean_squared_error', # label_heat
                'weight_stage4_L1': 'mean_squared_error', # label_vec
                'weight_stage4_L2': 'mean_squared_error', # label_heat
                'weight_stage5_L1': 'mean_squared_error', # label_vec
                'weight_stage5_L2': 'mean_squared_error', # label_heat
                'weight_stage6_L1': 'mean_squared_error', # label_vec
                'weight_stage6_L2': 'mean_squared_error'  # label_heat
            },
            metrics={
                'weight_stage1_L1': 'accuracy',
                'weight_stage1_L2': 'accuracy',
                'weight_stage2_L1': 'accuracy',
                'weight_stage2_L2': 'accuracy',
                'weight_stage3_L1': 'accuracy',
                'weight_stage3_L2': 'accuracy',
                'weight_stage4_L1': 'accuracy',
                'weight_stage4_L2': 'accuracy',
                'weight_stage5_L1': 'accuracy',
                'weight_stage5_L2': 'accuracy',
                'weight_stage6_L1': 'accuracy',
                'weight_stage6_L2': 'accuracy'
            }
        )
    else:
        model = Model(inputs=[input_tensor], outputs=[Mconv7_stage6_L1, Mconv7_stage6_L2])
        model.compile(optimizer=SGD(lr=lr, momentum=0.9, decay=0.0005, nesterov=True),
            loss={
                'Mconv7_stage6_L1': 'mean_squared_error', # label_vec
                'Mconv7_stage6_L2': 'mean_squared_error'  # label_heat
            },
            metrics={
                'Mconv7_stage6_L1': 'accuracy',
                'Mconv7_stage6_L2': 'accuracy'
            }
        )

    return model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/')
    parser.add_argument("--train", action='store_true', help='train')
    args = parser.parse_args()

    model = create_openpose(args.train)
    model.summary()
    model.save_weights("openpose.hdf5")
    with open('openpose.json', 'w') as f: f.write(model.to_json())

    from keras.utils import plot_model
    plot_model(model, to_file='openpose.png', show_shapes=True, show_layer_names=True)

    exit()
