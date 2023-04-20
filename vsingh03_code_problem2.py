import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def covariance_calc(a,b):
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    sum=0
    for i in range(len(a)):
        sum = sum+ (a[i]-a_mean)*(b[i]-b_mean)
    covariance = sum/len(a)
    return covariance    

def prob2_1(dataset):
    # Problem 2.1.1 Covariance matrix calculation
    data = np.loadtxt(dataset,delimiter=",", dtype=float)
    x,y,z = data[:,0],data[:,1],data[:,2]
    ax = plt.axes(projection='3d')
    ax.scatter3D(x,y,z,c=z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    title = "Visualized Dataset: " + dataset+" with surface normal direction"
    ax.set_title(title)

    # Calculating covariance
    covariance_matrix = np.zeros((3,3))
    covariance_matrix[0][0]=covariance_calc(x,x)
    covariance_matrix[0][1]=covariance_calc(x,y)
    covariance_matrix[0][2]=covariance_calc(x,z)
    covariance_matrix[1][0]=covariance_calc(y,x)
    covariance_matrix[1][1]=covariance_calc(y,y)
    covariance_matrix[1][2]=covariance_calc(y,z)
    covariance_matrix[2][0]=covariance_calc(z,x)
    covariance_matrix[2][1]=covariance_calc(z,y)
    covariance_matrix[2][2]=covariance_calc(z,z)
    print('Covariance Matrix is: \n',covariance_matrix)

    # Problem 2.1.2 Calculating Surface normal.
    print('\n Problem 2.1.2')
    vals, vects = np.linalg.eig(covariance_matrix)
    index = np.argmin(vals)
    coef = vects[:, index]
    print('Eig Values of the plane: ',vals)
    print('Eigen Vector corresponding to smallest eigen value: ',coef)
    print('Index selected: ', index)
    magnitude = coef[0]**2 +coef[1]**2 + coef[2]**2
    magnitude = np.sqrt(magnitude)
    print('Magnitude of surface normal =',magnitude)
    direction = np.arccos(coef)*180/np.pi
    print('Direction', direction)
    ax.quiver( np.mean(x), np.mean(y), np.mean(z),coef[0], coef[1], coef[2],color='blue',length=7.0)
    ax.text(-10, -10,-10, s="Surface Normal with increased magnitude for better visualization")

# Problem 2.2. Surface fitting using LS, TLS, and RANSAC
def leastsqcalc(x,y,z):
    o = np.ones(x.shape)
    t = np.vstack((x,y,o)).T
    t1 = np.dot(t.transpose() , t)
    t2 = np.dot(np.linalg.inv(t1), t.transpose())
    A = np.dot(t2, z.reshape(-1, 1))
    return A

def totallscalc(x,y,z, tls=False):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)
    arr = np.vstack(((x - x_mean), (y - y_mean), (z - z_mean))).T
    A = np.dot(arr.transpose(), arr)
    vals, vects = np.linalg.eig(A)
    index = np.argmin(vals)
    coef = vects[:, index]
    a, b, c = coef
    d = a * x_mean + b * y_mean + c * z_mean
    coef = np.array([a, b, c, d])
    if tls:
        print('\nEig Values: ',vals)
        print('Eigen Vectors: ',vects)
        print('Index selected: ',index)
    return coef

def ransac(x,y,z, threshold, outliers, p):
    size = x.shape[0]
    N_best = 0
    coef = np.zeros([3, 1])
    e = outliers / size
    s = 3
    iters = np.log(1 - p) / np.log(1 - np.power((1 - e), s))
    iters = np.maximum(int(iters), 50)
    print("\n Number of iterations for RANSAC = ", iters)

    for i in range(iters):
        # Selecting 3 points from given sample space
        random_indices = np.random.choice(size, size=3)
        x_random = x[random_indices]
        y_random = y[random_indices]
        z_random = z[random_indices]

        # Fit a model # Using total least squares for fitting the model.
        rand_coef = totallscalc(x_random,y_random,z_random)
        if np.any(np.iscomplex(rand_coef)):
            continue
        a, b, c, d = rand_coef
        Error = np.square((a * x) + (b * y) + (c * z) - d)

        #Checking if the given point is inlier
        for i in range(len(Error)):
            if float(Error[i]) > threshold:
                Error[i] = 0
            else:
                Error[i] = 1
        
        # Checking if current iteration is best fit
        N = np.sum(Error)
        if N > N_best:
            N_best = N
            coef = rand_coef
            x_fin,y_fin,z_fin = x_random,y_random,z_random
        
        if N_best/size >= p:
            break
    
    return coef, x_fin, y_fin, z_fin

def plotLSCurve(coef, x, y, z, ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_curve,y_curve = np.meshgrid(np.arange(xlim[0], xlim[1]),
                    np.arange(ylim[0], ylim[1]))
    z_curve = np.zeros(x_curve.shape)
    for r in range(x_curve.shape[0]):
        for c in range(x_curve.shape[1]):
            z_curve[r,c] = coef[0] * x_curve[r,c] + coef[1] * y_curve[r,c] + coef[2]
    ax.plot_wireframe(x_curve,y_curve,z_curve, color='r',label='Least Squares plane')
    return ax

def plotTotalLSCurve(coef, x, y, z, ax1, clr,label):
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    x_curve,y_curve = np.meshgrid(np.arange(xlim[0], xlim[1]),np.arange(ylim[0], ylim[1]))
    z_curve = np.zeros(x_curve.shape)
    for r in range(x_curve.shape[0]):
        for c in range(x_curve.shape[1]):
            z_curve[r,c] = coef[3] - (coef[0] * x_curve[r,c] + coef[1] * y_curve[r,c])
            z_curve[r,c] /= coef[2]
    ax1.plot_wireframe(x_curve,y_curve,z_curve, color = clr, label=label)
    return ax1

def problem2_2(dataset,outliers):
    # Reading dataset
    print('Reading Dataset: ',dataset)
    data = np.loadtxt(dataset,delimiter=",", dtype=float)
    x,y,z = data[:,0],data[:,1],data[:,2]

    #Plotting the data as scatter plot
    plt.figure()
    ax = plt.subplot(111, projection='3d')
    ax.scatter3D(x,y,z,c='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    title = "Different Plane fits for dataset " + dataset
    ax.set_title(title)

    # LS plane fit
    ls_coeff = leastsqcalc(x,y,z)
    print("\n LS Plane fit solution:")
    print("%f x + %f y + %f = z" % (ls_coeff[0], ls_coeff[1], ls_coeff[2]))
    #Plotting calculated plane and tracked points
    ax = plotLSCurve(ls_coeff, x, y, z, ax)
    
    # TLS plane fit
    tls_coeff = totallscalc(x,y,z,tls=True)
    print("\n Total LS Plane fit solution:")
    print("%f x + %f y + %f = z" % (-tls_coeff[0]/tls_coeff[2], -tls_coeff[1]/tls_coeff[2], tls_coeff[3]/tls_coeff[2]))
    #Plotting calculated plane and tracked points
    ax = plotTotalLSCurve(tls_coeff, x, y, z, ax, clr='g',label='Total Least Squares plane')

    # RANSAC plane fit
    threshold = 0.95
    # outliers_2 = 15
    p = 0.99
    ransac_coeff,x_fin,y_fin,z_fin = ransac(x,y,z, threshold, outliers, p)
    print('\n RANSAC Plane Fit Final Selected points')
    print(x_fin,y_fin,z_fin)
    print('\n RANSAC Plane Fit Solution')
    print("%f x + %f y + %f = z" % (-ransac_coeff[0]/ransac_coeff[2], -ransac_coeff[1]/ransac_coeff[2], ransac_coeff[3]/ransac_coeff[2]))
    #Plotting calculated plane and tracked points
    ax = plotTotalLSCurve(ransac_coeff,x_fin,y_fin,z_fin,ax,clr='b',label='RANSAC plane')
    ax.legend()

    
def main():
    dataset = "pc1.csv"
    print('\n')
    print('\nCurrent Dataset ',dataset)
    print('Solving Problem 2.1 for dataset: ',dataset)
    prob2_1(dataset)
    print('\n Solving Problem 2.2')
    problem2_2(dataset,outliers=1)
    
    dataset = "pc2.csv"
    print('\n')
    print('\nCurrent Dataset ',dataset)
    print('\nSolving Problem 2.2')
    problem2_2(dataset,outliers=15)
    # prob2_1(dataset)

    plt.show()

if __name__=='__main__':
    main()