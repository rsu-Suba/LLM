import tensorflow as tf

print("TensorFlow version:", tf.__version__)

gpu_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpu_devices))

if gpu_devices:
    print("GPU OK")
    for device in gpu_devices:
        print(device)
else:
    print("No GPU")