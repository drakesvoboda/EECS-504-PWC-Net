import cv2
import flow_vis
## Opencv Dense Flow - Gunner Farneback
def dense_GF(frame1, frame2, params=None, blur=True):
    if params is None:
        params = dict(pyr_scale = 0.5,
                      levels = 5,
                      winsize = 15,
                      iterations = 5, 
                      poly_n = 7, 
                      poly_sigma = 1.5,
                      flags = 0)
        
    prev_img = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    next_img = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    if blur:
        # prev_img = gaussian_blur(prev_img, 5, 1)
        # next_img = gaussian_blur(next_img, 5, 1)
        prev_img = cv2.GaussianBlur(prev_img,(5,5),0)
        next_img = cv2.GaussianBlur(next_img,(5,5),0)

    flow = cv2.calcOpticalFlowFarneback(prev_img,next_img, None, **params)
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)

    return flow, flow_color

# Dense Lucas-Kanade
def warp_flow_fast(im2, u, v):
    """ 
    im2 warped according to (u,v).
    This is a helper function that is used to warp an image to make it match another image 
    (in this case, I2 is warped to I1).
    Assumes im1[y, x] = im2[y + v[y, x], x + u[y, x]] 
    """
    # this code is confusing because we assume vx and vy are the negative
    # of where to send each pixel, as in the results by ce's siftflow code
    y, x = np.mgrid[:im2.shape[0], :im2.shape[1]]
    dy = (y + v).flatten()[np.newaxis, :]
    dx = (x + u).flatten()[np.newaxis, :]
    # this says: a recipe for making im1 is to make a new image where im[y, x] = im2[y + flow[y, x, 1], x + flow[y, x, 0]]
    return np.concatenate([scipy.ndimage.map_coordinates(im2[..., i], np.concatenate([dy, dx])).reshape(im2.shape[:2] + (1,)) \
                            for i in range(im2.shape[2])], axis = 2)

def gaussian_blur(img, kernel, sigma):
    kernel = cv2.getGaussianKernel(kernel, sigma)
    kernel = (kernel * kernel.T)
    img_ = img.copy()
    if img.ndim == 3:
        for i in range(3):  
            img_[:,:,i] = signal.correlate2d(img[:,:,i], kernel, 'same')
    else:
        img_[:,:] = signal.correlate2d(img[:,:], kernel, 'same')
    return img_

def pyr_down(p, kernel_size=3, sigma=0.8):
    '''
    Downsample the pyramid image to get the upper level. 
    Input:
      p: M x N x C array
    Return: 
      out: M/2 x N/2 x C 
    '''
    p_ = p.copy()
    out = gaussian_blur(p_, kernel_size, sigma)
    return cv2.resize(out, (int(p.shape[1]/2), int(p.shape[0]/2)), interpolation = cv2.INTER_CUBIC)

def buildGPyramid(im,nlevels):
    '''
    building pyramid of nlevels
    '''
    pyrd = []
    pyrd.append(im)
    for i in range(1,nlevels):
        pyrd.append(pyr_down(pyrd[-1]))
    return pyrd

def get_derivatives(I1,I2):
    '''
    Calculate Ixx, Iyy, Ixy, Ixt, Iyt. 
    '''
    # To increase robustness, over the method presented in the lecture notes:
    #      1. Blur the images using a Gaussian kernel to I1, I2 with kernel_size = 5, sigma = 1.
    #         before computing derivatives.
    #      2. Computing spatial derivatives Ix, Iy and It.
    #         For the spatial derivatives, please use the gradient filter [1, -8, 0, 8, -1]/12
    #      3. Get Ixx, Iyy, Ixy, Ixt, Iyt, then average over three channel 

    m, n, k = I1.shape
    Ixx = np.zeros((m, n))
    Iyy = np.zeros((m, n))
    Ixy = np.zeros((m, n))
    Ixt = np.zeros((m, n))
    Iyt = np.zeros((m, n))

    I1_blur = gaussian_blur(I1, 5, 1)
    I2_blur = gaussian_blur(I2, 5, 1)
    It = I2_blur - I1_blur
    
    # dx = np.array([[1, -8, 0, 8, -1]])/12
    dx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])/8
    dy = dx.T

    for i in range(k):
      Ix = signal.convolve2d(I1_blur[:,:,i], dx, mode='same')
      Iy = signal.convolve2d(I1_blur[:,:,i], dy, mode='same')
      
      Ixx = Ixx + Ix ** 2.0
      Iyy = Iyy + Iy ** 2.0
      Ixy = Ixy + Ix * Iy
      Ixt = Ixt + Ix * It[:,:,i]
      Iyt = Iyt + Iy * It[:,:,i]

    Ixx = Ixx/3.0
    Iyy = Iyy/3.0
    Ixy = Ixy/3.0
    Ixt = Ixt/3.0
    Iyt = Iyt/3.0

    return Ixx, Iyy, Ixy, Ixt, Iyt


def lucas_kanade(I1,I2,u,v,winsize,medfiltSize,nIterations):
    '''
    Lucas-Kanade algorithm. 
    Input:
        winsize - half the patch size, 
        medfiltSize - the size of the window for the spatial median filter,
        nIterations - the number of flow refinement iterations. warpI2 is the image I2 warped according to (u,v).
    Return: 
      out: M/2 x N/2 x C 
    '''
    # warp I2 according to (u,v)
    warpI2 = warp_flow_fast(I2,u,v)

    for i in range(nIterations):
        # compute derivatives
        Ixx, Iyy, Ixy, Ixt, Iyt = get_derivatives(I1,warpI2)

        # Sum the loss over every pixel in a small window.
        # Note that, unlike in class, we weigh the contribution of each pixel
        # based on its distance to the center using a Gaussian filter.
        a = gaussian_blur(Ixx, winsize*2+1, winsize/2.) + 0.001
        b = gaussian_blur(Ixy, winsize*2+1, winsize/2.)
        c = gaussian_blur(Iyy, winsize*2+1, winsize/2.) + 0.001
        d = gaussian_blur(Ixt, winsize*2+1, winsize/2.)
        e = gaussian_blur(Iyt, winsize*2+1, winsize/2.)

        # solve the 2x2 linear system at every pixel
        det = a * c - b**2. + np.finfo(float).eps
        du = -(c * d - b * e) / det
        dv = -(-b * d + a * e) / det

        # update the flow field and warp image
        u = u + du
        v = v + dv
        
        # median filtering
        if medfiltSize > 0:
            u = signal.medfilt2d(u, (medfiltSize, medfiltSize))
            v = signal.medfilt2d(v, (medfiltSize, medfiltSize))


        warpI2 = warp_flow_fast(I2, u, v)

    return u, v, warpI2

def coarse2fine_lk(im1,im2,nlevels,winsize,medfiltsize,nIterations):
    # building pyramid
    pyrd1 = buildGPyramid(im1,nlevels)
    pyrd2 = buildGPyramid(im2,nlevels)

    # compute from coarse level to fine level
    # Hint: Initialze u,v with all zeros
    u = np.zeros_like(pyrd1[nlevels-1][:,:, 0])
    v = np.zeros_like(pyrd1[nlevels-1][:,:, 0])

    for i in range(nlevels):
      I1 = pyrd1[nlevels-i-1]
      I2 = pyrd2[nlevels-i-1]

      u, v, warpI2 = lucas_kanade(I1,I2,u,v,winsize,medfiltsize,nIterations)

      if i != nlevels-1:
        u = cv2.resize(u, (u.shape[1]*2, u.shape[0]*2), interpolation = cv2.INTER_CUBIC)
        v = cv2.resize(v, (v.shape[1]*2, v.shape[0]*2), interpolation = cv2.INTER_CUBIC)
        u = 2.0 * u
        v = 2.0 * v
        u = gaussian_blur(u, 3, 0.8)
        v = gaussian_blur(v, 3, 0.8)

    return u, v, warpI2


def dense_LK(frame1, frame2):
    img1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img1_norm = cv2.normalize(img1.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    img2_norm = cv2.normalize(img2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    nlevels = 5
    winsize = 3
    medfiltsize = 11
    nIterations = 5

    u, v, warpI2 = coarse2fine_lk(img1_norm,img2_norm,nlevels,winsize,medfiltsize,nIterations)
    flow = np.array([u, v]).transpose(1,2,0)
    flow_color = flow_vis.flow_to_color(flow)
    return flow, flow_color