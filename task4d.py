#https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


cap = cv2.VideoCapture('colorbrighttest.mp4')

while(cap.isOpened()):
    # Take each frame
    ret, frame = cap.read()
    if ret:
    	res = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    	rows,cols,ch = frame.shape
    	pts1 = np.float32([[800,170],[1100,170],[800,850],[1100,850]])
    	pts2 = np.float32([[0,0],[1300,0],[0,900],[1300,900]])
    	
    	M = cv2.getPerspectiveTransform(pts1,pts2)

    	res = cv2.warpPerspective(frame,M,(1300,900))

    	cv2.imshow('res',res)

    	res = res.reshape((res.shape[0]*res.shape[1],3))
    	clt = KMeans(n_clusters=2)
    	clt.fit(res)

    	#res = cv2.rectangle(frame,(1100,170),(700,700),(50,50,255),2)
        
    	
    	hist = find_histogram(clt)
    	bar = plot_colors2(hist,clt.cluster_centers_)
    	plt.axis("off")
    	plt.imshow(bar)
    	plt.show()

    	k = cv2.waitKey(1) & 0xFF
    	if k == ord('q'):
            break


cap.release()
while(1):
    pass
#cv2.destroyAllWindows()