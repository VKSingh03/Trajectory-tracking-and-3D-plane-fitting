import cv2 as cv
import numpy as np
import imutils
from collections import deque
import argparse
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())
frame_size =[]

def filter(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower_red = np.array([0,100,100],np.uint8)
    upper_red = np.array([3,250,250],np.uint8)
    mask = cv.inRange(hsv,lower_red,upper_red)
    mask = cv.dilate(mask, None, iterations=2)
    result = cv.bitwise_and(frame,frame, mask=mask)
    result_gray = cv.cvtColor(result,cv.COLOR_BGR2GRAY)
    # print(result_gray.shape)
    # print(result.shape)
    # exit(0)
    # cv.imshow('Filtered Channel',result)
    # cv.imshow('Grayscale Channel',result_gray)
    ret, thresh = cv.threshold(result_gray,30,255, cv.THRESH_BINARY)
    return(thresh)

def scatter_plot(x,y):
    print('Plotting Tracked points')
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
    plt.xlabel('Horizontal Movement')
    plt.ylabel('Vertical Movement')
    plt.title('Trajectory of ball in actual scale')
    plt.ylim(562,0)
    plt.xlim(0,1218)

def image2plot(video):
    pts = deque(maxlen=args["buffer"])
    points = []
    while True:
        isTrue, frame = video.read() #isTrue stores boolean if the frame is imported properly
        
        if isTrue == True:
            cv.imshow('Frame',frame)
            frame = filter(frame)

            #Finding contours.
            cnts = cv.findContours(frame.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            if len(cnts) > 0:
                # finding largest contour based on area.
                # centroid constructed for that contour.
                c = max(cnts, key=cv.contourArea)
                ((x, y), radius) = cv.minEnclosingCircle(c)
                M = cv.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                if radius > 5:
                    # draw the enclosing circle and centroid
                    cv.circle(frame, (int(x), int(y)), int(radius),(255, 255, 255), 2)
                    cv.circle(frame, center, 5, (255, 255, 255), -1)
            # update the centre point
            pts.appendleft(center) 

            for i in range(1, len(pts)):
            # when Center is None, ignore them
                if pts[i - 1] is None or pts[i] is None: 
                    continue
                points.append(pts[i])

            # show the frame to screen
            cv.imshow("Trajectory Tracking", frame)
            frame_size = frame.shape #Prints the final frame of video, helps with pixels calculating. 
        else:
            break
        if cv.waitKey(20) & 0xFF==ord('c'):
            break
    video.release() # to delete the video pointer declared earlier
    cv.destroyAllWindows() # to close open windows 
    return points

def leastsqcalc(x,y):
    o = np.ones(x.shape)
    z = np.vstack((np.square(x), x, o)).T
    t1 = np.dot(z.transpose() , z)
    t2 = np.dot(np.linalg.inv(t1), z.transpose())
    A = np.dot(t2, y.reshape(-1, 1))
    return A

def plotLSCurve(coef, x, y):

    x_min = np.min(x)
    x_max = np.max(x)

    x_curve = np.linspace(x_min, x_max, 300) 
    o_curve = np.ones(x_curve.shape)
    z_curve = np.vstack((np.square(x_curve), x_curve, o_curve)).T
    y_curve = np.dot(z_curve, coef)

    plt.figure()
    plt.plot(x, y, 'ko', x_curve, y_curve, '-r')
    plt.xlabel('Horizontal Movement')
    plt.ylabel('Vertical Movement')
    plt.title('Least Squares Curve fitting')
    # ax = plt.axis()
    # plt.axis((ax[0],ax[1],ax[3],ax[2]))
    plt.ylim(562,0)
    plt.xlim(0,1218)

if __name__ == '__main__':
    
    # Reading the image
    video = cv.VideoCapture('ball.mov')

    # Getting object trajectory as points
    points = image2plot(video)       
    print('Size of frame:', frame_size)
    print('Starting point of ball in video frame', points[0])
    print('Final point of ball in video frame',points[-1])
    x,y = [],[]
    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    x = np.array(x) 
    y = np.array(y) 
    #Least squares calculation
    print('Calculating Best fit using Least Squares method \n')
    coefficient = leastsqcalc(x,y)
    print('Calculated Coefficients are: \n', coefficient,"\n")
    print(f'Equation of curve is: y = {coefficient[0]}*x^2 + {coefficient[1]}*x +{coefficient[2]} \n')
    #Plotting calculated line and tracked points
    scatter_plot(x,y)
    plotLSCurve(coefficient, x, y)
    
    pixel_x_final = 300*(x[-1]-x[0])/(y[-1]-y[0])
    print('X pixel of landing position of ball is: ', pixel_x_final)
    
    plt.show()