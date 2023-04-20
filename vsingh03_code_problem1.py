import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

frame_size =[]

def filter(frame):
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)#converting image to HSV color scale
    lower_red = np.array([0,100,100],np.uint8)
    upper_red = np.array([3,250,250],np.uint8)
    mask = cv.inRange(hsv,lower_red,upper_red) #creating red mask
    mask = cv.dilate(mask, None, iterations=2) #dilating mask for better result
    result = cv.bitwise_and(frame,frame, mask=mask) #applying mask on image
    result_gray = cv.cvtColor(result,cv.COLOR_BGR2GRAY) #converting masked image to grayscale
    ret, thresh = cv.threshold(result_gray,60,255, cv.THRESH_BINARY) #converting image to binary
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

def locate_center(frame):
    indices = np.where(frame==255)
    # Try except is used to avoid error when the ball is not present in video
    try:
        x = indices[1]
        y = indices[0]
        if(max(x)-min(x)<100) and (max(y)-min(y)<150):
            min_arg = np.argmin(y)
            max_arg = np.argmax(y)
            co_ordinate_min = np.array([x[min_arg], y[min_arg]])  
            co_ordinate_max = np.array([x[max_arg], y[max_arg]])
            co_ordinate = (co_ordinate_min + co_ordinate_max) / 2
            return co_ordinate
        else:
            return None
    except:
        return None


def image2plot(video):
    points = []
    pts = []
    while True:
        isTrue, frame = video.read() #isTrue stores boolean if the frame is imported properly
        
        if isTrue == True:
            cv.imshow('Frame',frame)
            frame = filter(frame)
            center = locate_center(frame) 
            pts.append(center) 
            # show the frame to screen 
            cv.imshow("Trajectory Tracking", frame) 
            # frame_size.append(frame.shape)
        else: 
            break 
        if cv.waitKey(20) & 0xFF==ord('c'): 
            break 
    
    # Cleaning the dataset for outliers as image thresholding is not perfect 
    for i in range(1, len(pts)): 
        # when Center is None, ignore them 
        if pts[i - 1] is None or pts[i] is None: 
            continue 
        # when the point is much far away from its neighbours, ignore them 
        elif((abs(pts[i-1][0] - pts[i][0])>20) or abs(pts[i-1][1] - pts[i][1])>15 ): 
            continue 
        points.append(pts[i]) 
        # print(pts[i][0], ',',pts[i][1]) 
    points = np.array(points) 
    print('Shape of points array', points.shape) 
    video.release() # to delete the video pointer declared earlier 
    cv.destroyAllWindows() # to close open windows 
    return points 

def leastsqcalc(x,y):
    o = np.ones(x.shape)
    z = np.vstack((np.square(x), x, o)).T
    t1 = np.dot(z.transpose() , z)
    t2 = np.dot(np.linalg.inv(t1), z.transpose())
    arr = np.dot(t2, y.reshape(-1, 1))
    return arr

def plotLSCurve(coef, x, y):

    x_min = np.min(x)
    x_max = np.max(x)

    x_curve = np.linspace(x_min-50, x_max+50, 300) 
    o_curve = np.ones(x_curve.shape)
    z_curve = np.vstack((np.square(x_curve), x_curve, o_curve)).T
    y_curve = np.dot(z_curve, coef)

    plt.figure()
    plt.plot(x, y, 'bo', x_curve, y_curve, '-r')
    plt.xlabel('Horizontal Movement')
    plt.ylabel('Vertical Movement')
    plt.title('Least Squares Curve fitting')
    # ax = plt.axis()
    # plt.axis((ax[0],ax[1],ax[3],ax[2]))
    plt.ylim(600,0)
    plt.xlim(-50,1300)

if __name__ == '__main__':

    # Reading the video
    video = cv.VideoCapture('ball.mov')

    # Getting object trajectory as points
    points = image2plot(video)       
    # print('Size of frame:', frame_size)
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
    # Plotting calculated line and tracked points
    scatter_plot(x,y)
    plotLSCurve(coefficient, x, y)

    pixel_x_final = 300*(x[-1]-x[0])/(y[-1]-y[0])
    print('X pixel of landing position of ball is: ', pixel_x_final)
    
    plt.show()