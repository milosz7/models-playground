from PIL import Image
import numpy as np
import time

def augment(img_arr: np.ndarray, crop_size: int, org_size: int):
    img = Image.fromarray(img_arr)
    if (np.random.random() < 0.5):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if(np.random.random() < 0.3333):
        crop_start = (org_size - crop_size) / 2
        rotation = 10 * np.random.random() - 5
        img = img.rotate(rotation, resample=Image.BILINEAR)
        img = img.crop((crop_start, crop_start, org_size - crop_start, org_size - crop_start))
    else:
        x = int((org_size - crop_size - 1) * np.random.random())
        y = int((org_size - crop_size - 1) * np.random.random())
        img = img.crop((x, y, x + crop_size, y + crop_size))
    return np.array(img)


def main():
    x: np.ndarray = np.load("data/parsed/cifar10/cifar10_training_images.npy")
    y: np.ndarray = np.load("data/parsed/cifar10/cifar10_training_labels.npy")
    nsets = 10
    new_dim = 28
    org_size = 32
    crop_start = (org_size - new_dim) / 2
    im = Image.fromarray(x[0,:])
    x_new = np.zeros((x.shape[0] * nsets, new_dim, new_dim, 3), dtype="uint8")
    y_new = np.zeros(y.shape[0] * nsets, dtype="uint8")

    t0 = time.time()
    
    k = 0
    for i in range(x.shape[0]):
        im = Image.fromarray(x[i,:])
        im = im.crop((crop_start, crop_start, org_size - crop_start, org_size - crop_start))
        x_new[k,...] = np.array(im)
        y_new[k] = y[i]
        k += 1
        for _ in range(nsets - 1):
            x_new[k,...] = augment(x[i,:], new_dim, org_size)
            y_new[k] = y[i]
            k += 1
    
    idx = np.argsort(np.random.random())
    x_new = x_new[idx]
    y_new = y_new[idx]

    np.save("data/parsed/cifar10/cifar10_training_images_extended.npy", x_new)
    np.save("data/parsed/cifar10/cifar10_training_labels_extended.npy", y_new)

    x = np.load("data/parsed/cifar10/cifar10_test_images.npy")
    x_new = np.zeros((x.shape[0], new_dim, new_dim, 3), dtype="uint8")
    
    for i in range(x.shape[0]):
        img = Image.fromarray(x[i,:])
        img = img.crop((crop_start, crop_start, org_size - crop_start, org_size - crop_start))
        x_new[i,...] = np.array(img)
    np.save("data/parsed/cifar10/cifar10_test_images_extended.npy", x_new)

    print("execution time:", time.time() - t0)


main()