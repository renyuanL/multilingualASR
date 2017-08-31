'''
ryFFTConv.py
因為 暫時安裝不起來 scipy 於 Office Computer 之 Python 3.4，
故駭進 D:\Anaconda3\Lib\site-packages\scipy\signal
找到 signaltools.py
挖出 fftconvolve()
再加上一些補丁，算是可以使用。

2016/08/01

'''


from numpy import (allclose, angle, arange, argsort, array, asarray,
                   atleast_1d, atleast_2d, cast, dot, exp, expand_dims,
                   iscomplexobj, mean, ndarray, newaxis, ones, pi,
                   poly, polyadd, polyder, polydiv, polymul, polysub, polyval,
                   prod, product, r_, ravel, real_if_close, reshape,
                   roots, sort, sum, take, transpose, unique, where, zeros,
                   zeros_like)
import numpy as np

_rfft_mt_safe =  True #(NumpyVersion(np.__version__) >= '1.9.0.dev-e24486e')

def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match

    
def fftconvolve(in1, in2, mode="full"):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`;
        if sizes of `in1` and `in2` are not equal then `in1` has to be the
        larger array.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Examples
    --------
    Autocorrelation of white noise is an impulse.  (This is at least 100 times
    as fast as `convolve`.)

    >>> from scipy import signal
    >>> sig = np.random.randn(1000)
    >>> autocorr = signal.fftconvolve(sig, sig[::-1], mode='full')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(np.arange(-len(sig)+1,len(sig)), autocorr)
    >>> ax_mag.set_title('Autocorrelation')
    >>> fig.tight_layout()
    >>> fig.show()

    Gaussian blur implemented using FFT convolution.  Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries.
    The `convolve2d` function allows for other types of image boundaries,
    but is far slower.

    >>> from scipy import misc
    >>> face = misc.face(gray=True)
    >>> kernel = np.outer(signal.gaussian(70, 8), signal.gaussian(70, 8))
    >>> blurred = signal.fftconvolve(face, kernel, mode='same')

    >>> fig, (ax_orig, ax_kernel, ax_blurred) = plt.subplots(1, 3)
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_kernel.imshow(kernel, cmap='gray')
    >>> ax_kernel.set_title('Gaussian kernel')
    >>> ax_kernel.set_axis_off()
    >>> ax_blurred.imshow(blurred, cmap='gray')
    >>> ax_blurred.set_title('Blurred')
    >>> ax_blurred.set_axis_off()
    >>> fig.show()

    """
    in1 = asarray(in1)
    in2 = asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return array([])

    s1 = array(in1.shape)
    s2 = array(in2.shape)
    complex_result = (np.issubdtype(in1.dtype, complex) or
                      np.issubdtype(in2.dtype, complex))
    shape = s1 + s2 - 1

    if mode == "valid":
        _check_valid_mode_shapes(s1, s2)

    # Speed up FFT by padding to optimal size for FFTPACK
    fshape = [_next_regular(int(d)) for d in shape]
    fslice = tuple([slice(0, int(sz)) for sz in shape])
    # Pre-1.9 NumPy FFT routines are not threadsafe.  For older NumPys, make
    # sure we only call rfftn/irfftn from one thread at a time.
    if not complex_result and (_rfft_mt_safe or _rfft_lock.acquire(False)):
        try:
            ret = (np.fft.irfftn(np.fft.rfftn(in1, fshape) *
                                 np.fft.rfftn(in2, fshape), fshape)[fslice].
                   copy())
        finally:
            if not _rfft_mt_safe:
                _rfft_lock.release()
    else:
        # If we're here, it's either because we need a complex result, or we
        # failed to acquire _rfft_lock (meaning rfftn isn't threadsafe and
        # is already in use by another thread).  In either case, use the
        # (threadsafe but slower) SciPy complex-FFT routines instead.
        ret = fftpack.ifftn(fftpack.fftn(in1, fshape) *
                            fftpack.fftn(in2, fshape))[fslice].copy()
        if not complex_result:
            ret = ret.real

    if mode == "full":
        return ret
    elif mode == "same":
        return _centered(ret, s1)
    elif mode == "valid":
        return _centered(ret, s1 - s2 + 1)
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

#fftconvolve([1,1,1,1],[1,1,1,1])


def blackmanharris(M, sym=True):
    """Return a minimum 4-term Blackman-Harris window.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.

    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).

    Examples
    --------
    Plot the window and its frequency response:

    >>> from scipy import signal
    >>> from scipy.fftpack import fft, fftshift
    >>> import matplotlib.pyplot as plt

    >>> window = signal.blackmanharris(51)
    >>> plt.plot(window)
    >>> plt.title("Blackman-Harris window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")

    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman-Harris window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")

    """
    if M < 1:
        return np.array([])
    if M == 1:
        return np.ones(1, 'd')
    odd = M % 2
    if not sym and not odd:
        M = M + 1
    a = [0.35875, 0.48829, 0.14128, 0.01168]
    n = np.arange(0, M)
    fac = n * 2 * np.pi / (M - 1.0)
    w = (a[0] - a[1] * np.cos(fac) +
         a[2] * np.cos(2 * fac) - a[3] * np.cos(3 * fac))
    if not sym and not odd:
        w = w[:-1]
    return w

