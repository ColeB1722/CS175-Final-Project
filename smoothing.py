import cv2
import numpy as np

##################
def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def L0Smoothing(Im, Lamb, Kap):
	betamax = 1e5
	fx = np.int32([[1, -1]])
	fy = np.int32([[1], [-1]])

	N, M, D = np.int32(Im.shape)
	S = np.float32(Im) / 256
	size2D = [N, M]

	otfFx = psf2otf(fx, size2D)
	otfFy = psf2otf(fy, size2D)

	# Compute F(I)
	FI = np.complex64(np.zeros((N, M, D)))
	FI[:,:,0] = np.fft.fft2(S[:,:,0])
	FI[:,:,1] = np.fft.fft2(S[:,:,1])
	FI[:,:,2] = np.fft.fft2(S[:,:,2])

	beta = 2 * Lamb

	MTF = np.power(np.abs(otfFx), 2) + np.power(np.abs(otfFy), 2)
	MTF = np.tile(MTF[:, :, np.newaxis], (1, 1, D))

	h = np.float32(np.zeros((N, M, D)))
	v = np.float32(np.zeros((N, M, D)))
	dxhp = np.float32(np.zeros((N, M, D)))
	dyvp = np.float32(np.zeros((N, M, D)))
	FS = np.complex64(np.zeros((N, M, D)))

	while beta < betamax:
		#h-v subproblem
		h[:,0:M-1,:] = np.diff(S, 1, 1)
		h[:,M-1:M,:] = S[:,0:1,:] - S[:,M-1:M,:]

    	# compute dySp
		v[0:N-1,:,:] = np.diff(S, 1, 0)
		v[N-1:N,:,:] = S[0:1,:,:] - S[N-1:N,:,:]

		if D == 1:
			t = np.power(h, 2) + np.power(v, 2) < lamb/beta
		else:
			t = np.sum(np.power(h, 2) + np.power(v, 2), axis=2) < Lamb / beta
			t = np.tile(t[:, :, np.newaxis], (1, 1, D))

		h[t] = 0
		v[t] = 0

    	#S subproblem
		dxhp[:,0:1,:] = h[:,M-1:M,:] - h[:,0:1,:]
		dxhp[:,1:M,:] = -(np.diff(h, 1, 1))
		dyvp[0:1,:,:] = v[N-1:N,:,:] - v[0:1,:,:]
		dyvp[1:N,:,:] = -(np.diff(v, 1, 0))
		normin = dxhp + dyvp

		FS[:,:,0] = np.fft.fft2(normin[:,:,0])
		FS[:,:,1] = np.fft.fft2(normin[:,:,1])
		FS[:,:,2] = np.fft.fft2(normin[:,:,2])

		denorm = 1 + beta * MTF;
		FS[:,:,:] = (FI + beta * FS) / denorm

		S[:,:,0] = np.float32((np.fft.ifft2(FS[:,:,0])).real)
		S[:,:,1] = np.float32((np.fft.ifft2(FS[:,:,1])).real)
		S[:,:,2] = np.float32((np.fft.ifft2(FS[:,:,2])).real)

		beta *= Kap

	S = S * 256

	print("inside", type(S))
	return S