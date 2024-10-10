import cv2
import numpy as np


def colorize_image(grayscale_image_path, output_image_path=None):
    # Load the pre-trained models from OpenCV
    prototxt = "model/colorization_deploy_v2.prototxt"
    caffemodel = "model/colorization_release_v2.caffemodel"
    pts = "model/pts_in_hull.npy"

    # Load the neural network
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    # Load cluster centers for ab channels (colorization model uses this)
    pts = np.load(pts)
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype(np.float32)]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, np.float32)]

    # Load and preprocess the grayscale image
    image = cv2.imread(grayscale_image_path)
    if image is None:
        raise ValueError(f"Could not load image from {grayscale_image_path}")

    # Convert the image to the LAB color space
    scaled = image.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize the L channel to the input size required by the network (224x224)
    l_channel = lab[:, :, 0]
    resized_l = cv2.resize(l_channel, (224, 224))
    resized_l = resized_l - 50  # Subtract the mean value for L channel

    # Prepare the network input and forward pass
    net.setInput(cv2.dnn.blobFromImage(resized_l))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize the predicted ab channels back to the original image size
    ab_channel = cv2.resize(ab_channel, (image.shape[1], image.shape[0]))

    # Merge the L channel with the predicted ab channels
    lab[:, :, 1:] = ab_channel

    # Convert the LAB image back to BGR color space
    colorized_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Clip values to the range [0, 255] and convert to uint8
    colorized_image = np.clip(colorized_image, 0, 1)
    colorized_image = (colorized_image * 255).astype(np.uint8)

    # Save the colorized image if an output path is provided
    if output_image_path:
        cv2.imwrite(output_image_path, colorized_image)

    return colorized_image