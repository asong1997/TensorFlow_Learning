import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets


# 设置TF的log等级为ERROR，也就是屏蔽通知和警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_val, y_val) = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)

print(x.shape, y.shape)
# 对训练数据集进行切片操作，每个batch = 200
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_dataset = train_dataset.batch(200)

# 构建神经网络层
# out = f(x @ w + b)
model = keras.Sequential([
    # [b, 784] @ [784, 512] + [b, 512] => [b, 512]
    layers.Dense(512, activation='relu'),
    # [b, 512] @ [512, 256] + [b, 256] => [b, 256]
    layers.Dense(256, activation='relu'),
    # [b, 256] @ [256, 10 ] + [b, 10 ] => [b, 10 ]
    layers.Dense(10)])

optimizer = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    # Step4.loop：每个epoch会有300个step: epoch_loop = shape(datasets) / batch = 60000 / 200 =300
    # 所以：每一个epoch执行300次step
    # 每个epoch指的是对整个数据集迭代一次
    # 每个step指的是对batch迭代一次
    for step, (x, y) in enumerate(train_dataset):

        with tf.GradientTape() as tape:
            # [b, 28, 28] => [b, 784]
            x = tf.reshape(x, (-1, 28 * 28))
            # Step1. compute output
            # [b, 784] => [b, 10]
            out = model(x)
            # Step2. compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]

        # Step3. optimize and update w1, w2, w3, b1, b2, b3
        # 损失函数loss对各个参数求导
        grads = tape.gradient(loss, model.trainable_variables)
        # 梯度更新：w' = w - lr * grad
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print("epoch={} step:{} loss={}".format((epoch+1), (step+100), loss.numpy()))

def train():
    # 对整个数据集迭代30次
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
