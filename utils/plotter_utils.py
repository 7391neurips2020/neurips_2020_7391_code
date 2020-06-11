import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_softmax(imgs, labels, preds, thresh=None):
    """Function to plot the prediction
    for a random input sample from a minibatch.
    :param imgs: Images in a minibatch
    :param labels: groundtruth contours for imgs
    :param preds: Predicted contour maps for imgs
    :param thresh: Threshold probability to count as
                    a contour pixel
    """
    viz = np.random.choice(range(len(imgs)), 1)
    plt.title('Plotting predictions for BSDS');
    plt.subplot(131);
    plt.imshow(imgs[viz][0]);
    plt.subplot(132);
    plt.imshow(labels[viz, :, :, 0][0]);
    prob_contours = np.exp(preds) / np.sum(np.exp(preds),
                                           axis=1,
                                           keepdims=True)
    prob_contours = prob_contours[viz, :, :, 1]
    prob_contours[prob_contours >= prob_contours.mean()] = 1.
    prob_contours[prob_contours != 1.] = 0.
    prob_contours = prob_contours[0]
    plt.subplot(133);
    plt.imshow(prob_contours,
               vmin=prob_contours.min(),
               vmax=prob_contours.max());
    plt.show()


def plot_prediction_softmax(self, imgs, labels, preds, thresh=None):
    """Function to plot the prediction
    for a random input sample from a minibatch.
    :param imgs: Images in a minibatch
    :param labels: groundtruth contours for imgs
    :param preds: Predicted contour maps for imgs
    :param thresh: Threshold probability to count as
                    a contour pixel
    """
    viz = np.random.choice(range(len(imgs)),1)
    plt.title('Plotting predictions for BSDS');
    plt.subplot(131); plt.imshow(imgs[viz][0]);
    plt.subplot(132); plt.imshow(labels[viz,:,:,0][0]);
    prob_contours = np.exp(preds)/np.sum(np.exp(preds),
                                        axis=1,
                                        keepdims=True)
    prob_contours = prob_contours[viz,:,:,1]
    prob_contours[prob_contours>=prob_contours.mean()] = 1.
    prob_contours[prob_contours!=1.] = 0.
    prob_contours = prob_contours[0]
    plt.subplot(133); plt.imshow(prob_contours,
                        vmin=prob_contours.min(),
                        vmax=prob_contours.max());
    plt.show()