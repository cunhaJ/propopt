####propagate.py   


def fresnel(z, mask, npixmask, pixsizemask, npixscreen, dxscreen, dyscreen, wavelength):
    import numpy as np
    import scipy.fftpack as sfft  
    
    k = 2* np.pi/wavelength
    
    #number of pixels 
    nps = npixscreen 
    npm = npixmask 
    
    ##calculate the resolution  
    res = wavelength *z/ (pixsizemask *npm)
    
    #dmask = npixmask * npm 
    dmask = 0.5 * res * npm
    
    xm1 = np.linspace(-dmask/2, dmask/2, npm)
    ym1 = np.linspace(-dmask/2, dmask/2, npm)
    (xm, ym) = np.meshgrid(xm1, ym1)

    xs1 = np.linspace(-dxscreen, dxscreen, nps)
    ys1 = np.linspace(-dyscreen, dyscreen, nps)
    (xs, ys) = np.meshgrid(xs1, ys1)
    
    #Goodman, exp 4-17
    v1  = np.exp(1.0j*k* (xm*xm + ym*ym)/ (2*z))
    v2  = np.exp(1.0j*k* (xs*xs + ys*ys)/ (2*z)) 
    v3  = np.exp(1.0j*k*z)/ (1.0j*wavelength*z)
    intarg = v1 * mask
    Ef = v2 * v3 * sfft.fftshift(sfft.fft2(sfft.ifftshift(intarg)))

    
    return abs(Ef)**2
    

def fraunhofer(z, mask, npixmask, pixsizemask, npixscreen, dxscreen, dyscreen, wavelength):
    import numpy as np
    import scipy.fftpack as sfft    
    
    k = 2* np.pi/wavelength
    #number of pixels 
    nps = npixscreen
    npm = npixmask 
    
    dmask = npixmask * npm  
    
    xm1 = np.linspace(-dmask/2, dmask/2, npm)
    ym1 = np.linspace(-dmask/2, dmask/2, npm)
    (xm, ym) = np.meshgrid(xm1, ym1)
    
    xs1 = np.linspace(-dxscreen, dxscreen, nps)
    ys1 = np.linspace(-dyscreen, dyscreen, nps)
    (xs, ys) = np.meshgrid(xs1, ys1)
    
    
    delta1=pixsizemask

    #delta2 = z*wavelength /(npm*delta1)
    #num = np.round(dxscreen/delta2)
    #print(num)
    
    #import cv2 
    #resized = cv2.resize(mask,(int(4*num),int(4*num)), interpolation = cv2.INTER_AREA)
    resized = mask
    
    #Goodman, exp 4-25
    v2  = np.exp(1.0j*k* (xs*xs + ys*ys)/ (2*z)) 
    v3  = np.exp(1.0j*k*z)/ (1.0j*wavelength*z)
    Ef = v2 * v3* sfft.fftshift(sfft.fft2(resized))
    

    #print(resized.shape)
    
    #Ef2 = cv2.resize(Ef,(int(num),int(num)), interpolation = cv2.INTER_AREA)
    
    return np.abs(Ef)**2
    
def RS_int(zs, mask, npixmask, pixsizemask, npixscreen, dxscreen, dyscreen, wavelength, I0, verbose =False ): 
    """
    returns Escreen (complex electric field at obs screen), Iscreen (intensity at obs screen), iplot (the actual intensity) 
    inputs: 
    zs = distance to screen [m]
    mask = image of the mask (normalized to [0,1]) - in principle could be grayscale between 0 and 1 
    npixmask = number of pixels on the side of the mask 
    pixsizemask = size of pixel of the mask [m]
    npixscreen = number of pixels on the side of the screen 
    dxscreen = max_x of the screen [m], the screen range is [-dxscreen, dxscreen]
    dyscreen = max_y of the screen [m], the screen range is [-dyscreen, dyscreen]
    wavelength = wavelength of the light [m]
    I0 = intensity of the light at the mask plane [W/m2]
    
    ------- 
    optional 
    verbose, defaults to False, if True prints 
    """
    import decimal
    import numpy as np 
    import matplotlib.pyplot as plt 
    
    # set the precision to double that of float64.. or whatever you want.
    decimal.setcontext(decimal.Context(prec=34))

    #number of pixels 
    nps = npixscreen 
    npm = npixmask 
    
    if nps <= 2*npm: 
        print("The number of screen is not large enough, I will resize them for you.")
        nps = npm*4 
        print("Rescaled the screen pixels to "+str(nps)+" . The computation will now proceed")
    
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
        if verbose == True: 
            print(isc/nps)
            
        for jsc in np.arange(0,nps-1): 
            r = np.sqrt((xs[isc,jsc]-xm)**2 + (ys[isc,jsc]-ym)**2 + (zs-zm)**2)
            r2 = r*r
            prop1= np.exp(-r*1.0j*k)/r2
            prop2 = zs * (1.0j * k  + unit/r)
            propE = E0m * prop1 * prop2
            #here npm*400, is a guess for the number of points to calc the int
            rEs[isc,jsc] = double_Integral(-dmask/2, dmask/2, -dmask/2, dmask/2, npm*400,npm*400,np.real(propE))/(2*np.pi)
            iEs[isc,jsc] = double_Integral(-dmask/2, dmask/2, -dmask/2, dmask/2, npm*400,npm*400,np.imag(propE))/(2*np.pi)

    Escreen = rEs + 1.0j*iEs 
    Iscreen = (c_const*n*eps0/2) * np.abs(Escreen)**2
    iplot = 10*Iscreen**0.2
    iplotmax = np.max(iplot)
    
    return Escreen, Iscreen, iplot 
    
    
#rudimentary 2D integral, following https://stackoverflow.com/questions/20668689/integrating-2d-samples-on-a-rectangular-grid-using-scipy 
def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):
    import numpy as np 

    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    A_Internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (np.sum(A_Internal)\
                + 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))\
                + 0.25 * (A_ul + A_ur + A_dl + A_dr))

###These are the theoretical functions as extracted from Goodman book 

def circ_fraun(aperture_rad, rcoord, zdist, wavelength): 
    #circular aperture function 4.4.2
    import numpy as np
    from scipy.special import jv 
    ###use like jv(v,z) where v is the order and z the dist in z 
    
    area = np.pi * aperture_rad**2
    
    k = 2* np.pi/wavelength
    argument = k*aperture_rad*rcoord/zdist
    
    #function 4-31
    intensity = (area/(wavelength * zdist))**2 * (2*jv(1, argument)/argument)**2
    
    return intensity 
    
    
def rect_fraun(sizex, sizey, xcoord, ycoord, zdist, wavelength): 
    from scipy.special import sinc 
    #sinc uses sin(pi*x)/(pi*x) with x as the argument 
    
    area = 4*sizex*sizey
    
    k= 2* np.pi/wavelength 
    
    intensity = (area/(wavelength*zdist))**2 * sinc(2*sizex*xcoord/(wavelength*zdist))**2 * sinc(2*sizey * ycoord/(wavelength* zdist))
    
    return intensity 
    
###This function is the axial intensity for a circular aperture following 1992 Sheppard paper, exp (28) 
def circ_zz(aperture_rad, zdist, wavelength):
    k = 2*np.pi /wavelength 
    
    izz1 = (1 + np.sqrt(1 + aperture_rad**2 /zdist**2))
    izz2 = 1 + aperture_rad**2/(2*zdist**2)
    izz3 = (k * aperture_rad**2/(2 *zdist))/(np.sqrt(1+aperture_rad**2/zdist**2)+1)
    itot = 0.25*(izz1/izz2) * np.sin(izz3)**2
    
    return itot

###This function is the axial intensity for a circular aperture following 1992 Sheppard paper, exp (28) 
def circ_zz24(aperture_rad, zdist, wavelength):
    k = 2*np.pi /wavelength 
    
    izz1 = 1/(1+aperture_rad**2/zdist**2)
    izz2 = 2/np.sqrt(1+aperture_rad**2/zdist**2)
    izz3 = (k * aperture_rad**2/(zdist)) /(np.sqrt(1+aperture_rad**2/zdist**2)+1)
    itot = 0.25*(1+izz1-izz2)* np.cos(izz3)
    
    return itot


#######FUNCTION THAT CALCULATES THE RS INTEGRAL 
#here is a function that calculates the RS_int of the first kind, taking information about mask, distance to screen, and screen information
def RS_int_XXZZ(zs, nzds, mask, npixmask, pixsizemask, npixscreen, dxscreen, dyscreen, wavelength, I0, verbose=False, logscale = False): 
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
    # set the precision to double that of float64
    decimal.setcontext(decimal.Context(prec=34))

    #number of pixels 
    nps = npixscreen 
    npm = npixmask 
    
    
    
    if nps <= 2*npm: 
        print("The number of screen is not large enough, I will resize them for you.")
        nps = npm*4 
        print("Rescaled the screen pixels to "+str(nps)+" . The computation will now proceed")
    
    #size of mask 
    dmask = pixsizemask * npm
    
    #physical constants 
    c_const = 3e8 #m/s
    eps0 = 8.85e-12  #F/m
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
    
    print(ys[0,:])
    
    
    #array with the distance in z until the distance zs given as argument 
    zdistarray = np.linspace(0,zs,nzds)
    
    if logscale ==True: 
        zdistarray = np.logspace(np.log(1e-6),np.log(zs),nzds, base=np.e)
    
    inten = np.zeros((len(xs1), len(zdistarray)))
    
    print (inten)

    ###### calculate the Rayleigh Sommerfeld integral 
    ### ATTENTION TO CHECK WITH GOODMAN BOOK IF CORRECTLY IMPLEMENTED 
    
    for iz, zsd in enumerate(zdistarray): 
        ##### calculate the Rayleigh Sommerfeld integral 
        #From Oshea formulation, eq 2.8
        for isc in np.arange(0,nps-1):
            if verbose == True: 
                print(isc/nps)
                
            for jsc in np.arange(0,nps-1): 
                r = np.sqrt((xs[isc,jsc]-xm)**2 + (ys[isc,jsc]-ym)**2 + (zsd-zm)**2)
                r2 = r*r
                prop1= np.exp(-r*1.0j*k)/r2
                prop2 = zsd * (1.0j * k  + unit/r)
                propE = E0m * prop1 * prop2
                #here npm*400, is a guess for the number of points to calc the int
                rEs[isc,jsc] = double_Integral(-dmask/2, dmask/2, -dmask/2, dmask/2, npm*400,npm*400,np.real(propE))/(2*np.pi)
                iEs[isc,jsc] = double_Integral(-dmask/2, dmask/2, -dmask/2, dmask/2, npm*400,npm*400,np.imag(propE))/(2*np.pi)

        Escreen = rEs + 1.0j*iEs 
        Iscreen = (c_const*n*eps0/2) * np.abs(Escreen)**2
        iplot = 10*Iscreen**0.3
        #print(iplot)
        midpoint = int(nps/2)-1
        inten[:,iz] = iplot[:,midpoint]
        #inten[isc,iz] = iplot
        
        print(inten[:,iz])
        
        iplotmax = np.max(iplot)
    
    return Escreen, Iscreen, inten
    
    
def Fresnel_num(width, wavelength, zdist):
    """
    Following the Goodman pag 85
    width is half the size of the aperture
    
    The Fresnel approximation is justified for large fresnel numbers 
    """
    NF = width**2 / (wavelength * zdist)
    return NF 
    
def Fraunhofer_criterion(aperturesiz, wavelength): 
    zfraun = 2 * aperturesiz**2 / wavelength 
    
    return zfraun 
