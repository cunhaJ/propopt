####propagate.py   
    
#######FUNCTION THAT CALCULATES THE RS INTEGRAL PROPAGATOR XY

def fresnel(z, mask, npixmask, pixsizemask, npixscreen, dxscreen, dyscreen, wavelength):
    k = 2* np.pi/wavelength
    
    #number of pixels 
    nps = npixscreen 
    npm = npixmask 
    
    #if nps <= 2*npm: 
    #    print("The number of screen is not large enough, I will resize them for you.")
    #    nps = npm*4 
    #    print("Rescaled the screen pixels to "+str(nps)+"^2 . The computation will now proceed")
    
    dmask = npixmask * npm 
    
    xm1 = np.linspace(-dmask/2, dmask/2, npm)
    ym1 = np.linspace(-dmask/2, dmask/2, npm)
    (xm, ym) = np.meshgrid(xm1, ym1)
     
    xs1 = np.linspace(-dxscreen, dxscreen, nps)
    ys1 = np.linspace(-dyscreen, dyscreen, nps)
    (xs, ys) = np.meshgrid(xs1, ys1)
    
    v1  = np.exp(1.0j*k* (xm*xm + ym*ym)/ (2*z))
    v2  = np.exp(1.0j*k* (xs*xs + ys*ys)/ (2*z)) 
    v3  = np.exp(1.0j*k*z)/ (1.0j*wavelength*z)
    U = v2 * v3 * np.fft.fftshift(np.fft.fft2(v1*mask))
    
    return np.abs(U)**2



def fraunhofer(z, mask, npixmask, pixsizemask, npixscreen, dxscreen, dyscreen, wavelength):
    k = 2* np.pi/wavelength
    #number of pixels 
    nps = npixscreen
    npm = npixmask 
    
    
    dmask = npixmask * npm  
    
    xm1 = np.linspace(-dmask/2, dmask/2, npm)
    ym1 = np.linspace(-dmask/2, dmask/2, npm)
    (xm, ym) = np.meshgrid(xm1, ym1)
    
    zdist = z 
    wavelength = wavelength 
    delta1=pixsizemask

    delta2 = zdist*wavelength /(npm*delta1)
    num = np.round(dxscreen/delta2)
    #print(num)
    
    import cv2 
    resized = cv2.resize(mask,(int(num),int(num)), interpolation = cv2.INTER_AREA)
    #print(resized.shape)
    
    v = np.exp(1.0j*k*z) / (1.0j*wavelength*z)
    U = v * np.fft.fftshift(np.fft.fft2(resized))
    return np.abs(U)**2
    
def RS_intv2(zs, mask, npixmask, pixsizemask, npixscreen, dxscreen, dyscreen, wavelength, I0): 
    """
    returns Escreen (complex electric field at obs screen), Iscreen (intensity at obs screen), iplot (the actual intensity) 
    inputs: 
    zs = distance to screen [m]
    mask = image of the mask (normalized to [0,1]) - in principle could be grayscale between 0 and 1 
    npixmask = number of pixels on the side of the mask 
    pixsizemask = size of pixel of the mask [m]
    npixscreen = number of pixels on the side of the screen 
    dxscreen = x side of the screen [m]
    dyscreen = y side of the screen [m]
    wavelength = wavelength of the light [m]
    I0 = intensity of the light at the mask plane [W/m2]
    """
    import decimal
    # set the precision to double that of float64.. or whatever you want.
    decimal.setcontext(decimal.Context(prec=34))

    #number of pixels 
    nps = npixscreen 
    npm = npixmask 
    
    if nps <= 2*npm: 
        print("The number of screen is not large enough, I will resize them for you.")
        nps = npm*4 
        print("Rescaled the screen pixels to "+str(nps)+"^2 . The computation will now proceed")
    
    #size of mask 
    dmask = pixsizemask * npm
    
    #physical constants 
    c_const = 3e8 
    eps0 = 8.854189e-12 
    n = 1 #refractive index of medium  

    k = 2* np.pi/wavelength

    ## definitions 
    unit = np.ones((npm,npm), dtype=complex)
    r = np.zeros((npm,npm)) 
    r3 = np.zeros((npm,npm))
    prop1 = np.zeros((npm,npm))
    prop2 = np.zeros((npm,npm))
    propE = np.zeros((npm,npm))
    
    #electric field real and imaginary and total 
    rEs =np.zeros ((nps,nps))
    iEs =np.zeros ((nps,nps))
    Escreen =np.zeros ((nps,nps), dtype =complex)

    #define the zpos of the mask at 0 
    zm =0 

    #definition of the mask
    xm1 = np.linspace(-dmask/2, dmask/2, npm)
    ym1 = np.linspace(-dmask/2, dmask/2, npm)
    (xm, ym) = np.meshgrid(xm1,ym1)
    
    ##Electric field calc from intensity
    E0 = np.sqrt(2*I0/(c_const*n*eps0))
    
    #definition of the electric field at the mask 
    E0m = E0 * mask

    fig=plt.figure()
    plt.imshow(E0m, vmin=0, vmax=1, cmap=plt.get_cmap("jet"))
    plt.title("Efield at mask")

    #intensity at the mask
    i0m = (c_const*n*eps0/2)*E0m*E0m 
    I0m = double_Integral(-dmask/2, dmask/2, -dmask/2, dmask/2, npm,npm, i0m)

    print(I0m)

    Utheoreticalmax = I0 * np.pi * dmask**2 
    print(Utheoreticalmax)

    xs1 = np.linspace(-dxscreen, dxscreen, nps)
    ys1 = np.linspace(-dyscreen, dyscreen, nps)
    (xs, ys) = np.meshgrid(xs1,ys1)

    ###### calculate the Rayleigh Sommerfeld integral 
    #From Oshea formulation, eq 2.8
    for isc in np.arange(0,nps-1):
        print(isc/nps)
        for jsc in np.arange(0,nps-1): 
            r = np.sqrt((xs[isc,jsc]-xm)**2 + (ys[isc,jsc]-ym)**2 + (zs-zm)**2)
            r2 = r*r
            prop1= np.exp(-r*1.0j*k)/r2
            prop2 = zs * (1.0j * k  + unit/r)
            propE = E0m * prop1 * prop2
            rEs[isc,jsc] = double_Integral(-dmask/2, dmask/2, -dmask/2, dmask/2, npm*100,npm*100,np.real(propE))/(2*np.pi)
            iEs[isc,jsc] = double_Integral(-dmask/2, dmask/2, -dmask/2, dmask/2, npm*100,npm*100,np.imag(propE))/(2*np.pi)

    Escreen = rEs + 1.0j*iEs 
    Iscreen = (c_const*n*eps0/2) * np.abs(Escreen)**2
    iplot = 10*Iscreen**0.2
    iplotmax = np.max(iplot)
    
    return Escreen, Iscreen, iplot 
    
#rudimentary 2D intergal, following https://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy 
def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):

    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    A_Internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (np.sum(A_Internal)\
                + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))\
                + 0.25 * (A_ul + A_ur + A_dl + A_dr))

#######FUNCTION THAT CALCULATES THE RS INTEGRAL PROPAGATOR XZ 

#######FUNCTION THAT CALCULATES THE RS CONV. PROPAGATOR CO

#######FUNCTION THAT CALCULATES THE FRESNEL APPROX.INTEGRAL 

#######FUNCTION THAT CALCULATES THE FRAUNHOFER APPROX. INTEGRAL