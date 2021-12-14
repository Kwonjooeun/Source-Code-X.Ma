from keras.layers import Input, Dense, BatchNormalization, Conv2D, LeakyReLU, Lambda, Flatten, Reshape
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import h5py
import numpy as np
import math
import cmath
import time
from utils import *
import os
import scipy.io
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num = 2
m = 16
n = 16
N_BS = m*n
N_ms = 1
cr = 1/2
mtx_v = N_BS
mtx_h = int(N_BS*cr)
output_dim = int(N_BS*cr)

# snr=0, nmse=-6.1681, cr=1/2
# snr=0, nmse=-3.57854, cr=0.25
SNR = [0]
# data loading
train = 'H_train.mat'
val = 'H_val.mat'
test = 'H_test.mat'

mat = scipy.io.loadmat(train)
x_train = mat['H_train']
# x_train = np.transpose(x_train, [2, 1, 0])
x_train = x_train.astype('float32')  # 训练变量类型转换

mat = scipy.io.loadmat(val)
x_val = mat['H_val']
# x_val = np.transpose(x_val, [2, 1, 0])
x_val = x_val.astype('float32')  # 训练变量类型转换

mat = scipy.io.loadmat(test)
x_test = mat['H_test']
# x_test = np.transpose(x_test, [2, 1, 0])
x_test = x_test.astype('float32')  # 训练变量类型转换


def network(inputs, snr):
    y_tmp = Dense(output_dim, activation='linear', use_bias=False)(inputs)
    # y_tmp = MatMul(output_dim, mtx_v, mtx_h)(inputs)
    y = Lambda(get_noise, arguments={'snr': snr})(y_tmp)
    decode = Dense(N_BS, activation='linear', use_bias=False)(y)
    # decode = MatMul(dictionary_dim, dictionary_mtx_v, dictionary_mtx_h)(y)
    h_tmp = Reshape((num, m, n))(decode)
    h = residual_block_decoded(h_tmp)
    h = Reshape((num, N_BS))(h)
    # h = Dense(N_BS, activation='linear')(h)
    # h = Lambda(dic, arguments={'P_real_init': P_real, 'P_imag_init':P_imag})(h)
    return h


for snr in SNR:
    inpt = Input(shape=(num, N_BS,))
    outpt = network(inpt, snr)
    model = Model(inputs=[inpt], outputs=[outpt])
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses_train = []
            self.losses_val = []

        def on_batch_end(self, batch, logs={}):
            self.losses_train.append(logs.get('loss'))

        def on_epoch_end(self, epoch, logs={}):
            self.losses_val.append(logs.get('val_loss'))


    history = LossHistory()
    file = 'model3' + '_cr_' + str(cr) + '_snr_' + str(snr)
    path = 'result_25/TensorBoard_%s' % file

    model.fit(x_train, x_train,
              epochs=300,
              batch_size=128,
              shuffle=True,
              validation_data=(x_val, x_val),
              callbacks=[history,
                         TensorBoard(log_dir=path)])

    tStart = time.time()
    x_hat = model.predict(x_test)
    tEnd = time.time()
    print("It cost %f sec" % ((tEnd - tStart) / x_test.shape[0]))

    x_re = x_test[:, 0, :]
    x_im = x_test[:, 1, :]
    x_test_c = x_re + 1j * x_im

    x_hat_re = x_hat[:, 0, :]
    x_hat_im = x_hat[:, 1, :]
    x_hat_c = x_hat_re + 1j * x_hat_im

    power = np.sum(abs(x_test_c) ** 2, axis=1)
    mse = np.sum(abs(x_test_c - x_hat_c) ** 2, axis=1)

    print("When compress rate is", cr)
    print('SNR is', snr, 'dB')
    print("NMSE is ", 10 * math.log10(np.mean(mse / power)), 'dB')
    '''
    filename = "result/model_%s.csv" % file
    x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
    np.savetxt(filename, x_hat1, delimiter=",")
    '''
    model_json = model.to_json()
    outfile = "result_25/saved_model/%s.json" % file
    with open(outfile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    outfile = "result_25/saved_model/%s.h5" % file
    model.save_weights(outfile)

# NMSE = []
# SNR = [0, 5, 10, 15, 20]
# for snr in SNR:
# #     test = 'test_snr_' + str(snr) + '_cr_' + str(cr) + '.mat'
# #     mat = h5py.File(test)
# #     x_test = mat['x_test']
# #     x_test = np.transpose(x_test, [2, 1, 0])
# #     x_test = x_test.astype('float32')  # 训练变量类型转换
# #     # test = 'old_375/H_test.mat'
#     # mat = h5py.File(test)
#     # x_test = mat['H_test']
#     # x_test = np.transpose(x_test, [2, 1, 0])
#     # x_test = x_test.astype('float32')  # 训练变量类型转换

#     file = 'model_new_D' + '_cr_' + str(cr) + '_snr_' + str(snr)
#     outfile = "result_5/saved_model/%s.json" % file
#     # outfile = "old_375/%s.json" % file
#     json_file = open(outfile, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     net = model_from_json(loaded_model_json, custom_objects={'k': k})
#     outfile = "result_5/saved_model/%s.h5" % file
#     # outfile = "old_375/%s.h5" % file
#     net.load_weights(outfile)

#     x_hat = net.predict(x_test)

#     x_re = x_test[:, 0, :]
#     x_im = x_test[:, 1, :]
#     x_test_c = x_re + 1j * x_im

#     x_hat_re = x_hat[:, 0, :]
#     x_hat_im = x_hat[:, 1, :]
#     x_hat_c = x_hat_re + 1j * x_hat_im

#     power = np.sum(abs(x_test_c) ** 2, axis=1)
#     mse = np.sum(abs(x_test_c - x_hat_c) ** 2, axis=1)
#     nmse = 10*math.log10(np.mean(mse/power))
#     print('SNR is', snr, 'dB')
#     print("NMSE is ", nmse, 'dB')
#     NMSE.append(nmse)

# sio.savemat('NMSE_DL_4_path_cr_'+ str(cr) + '.mat', mdict={'NMSE_DL_4_path_2': NMSE})