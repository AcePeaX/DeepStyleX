import matplotlib.pyplot as plt
import numpy as np

def imshow(images, titles=[""], callback = None, nrows = 0, ncols=0):
    """Plot a multiple images with titles.

    Parameters
    ----------
    images : image list
    titles : title list
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    if type(images)!=list or type(images)!=tuple:
        images = [images]
    if ncols == 0 and nrows == 0:
        ncols = len(images)
        nrows = 1
    if ncols == 0:
      ncols = len(images) // nrows
    if nrows == 0:
      nrows = len(images) // ncols

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, squeeze=False, figsize = None)
    for i, image in enumerate(images):
        axeslist.ravel()[i].imshow(image, cmap=plt.gray(), vmin=0, vmax=255)
        axeslist.ravel()[i].set_title(titles[i])
        axeslist.ravel()[i].set_axis_off()
    plt.tight_layout() # optional

    text=fig.text(0,0, "", va="bottom", ha="left")
    def onclick(event):
        [i],[j] = np.where(axeslist == event.inaxes)
        callback(axeslist, [i,j], [event.xdata, event.ydata], text)
    
    # Create an hard reference to the callback not to be cleared by the garbage collector
    ka = fig.canvas.mpl_connect('button_press_event', onclick)
    return axeslist
