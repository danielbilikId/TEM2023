#!/usr/bin/env python
import numpy as np

"""
Time encoding and decoding algorithms that make use of the
integrate-and-fire neuron model.

- iaf_decode            - IAF time decoding machine.
- iaf_decode_fast       - Fast IAF time decoding machine.
- iaf_decode_pop        - MISO IAF time decoding machine.
- iaf decode_coupled    - MISO coupled IAF time decoding machine.
- iaf_decode_delay      - MIMO delayed IAF time decoding machine.
- iaf_decode_spline     - Spline interpolation IAF time decoding machine.
- iaf_decode_spline_pop - MISO spline interpolation IAF time decoding machine.
- iaf_encode            - IAF time encoding machine.
- iaf_encode_delay      - MIMO delayed IAF time decoding machine.
- iaf_encode_coupled    - SIMO coupled IAF time encoding machine.
- iaf_encode_pop        - MIMO IAF time encoding machine.
- iaf_recoverable       - IAF time encoding parameter check.
"""

# Copyright (c) 2009-2015, Lev Givon
# All rights reserved.
# Distributed under the terms of the BSD license:
# http://www.opensource.org/licenses/bsd-license

__all__ = ['iaf_recoverable', 'iaf_encode', 'iaf_decode']

import numpy as np
import scipy.signal
import scipy.integrate

# The sici() and ei() functions are used to construct the matrix G in
# certain decoding algorithms because they can respectively compute
# the sine and exponential integrals relatively quickly:
import scipy.special

import numpy_extras as ne
import scipy_extras as se

# Pseudoinverse singular value cutoff:
__pinv_rcond__ = 1e-8


def iaf_recoverable(u, bw, b, d, R, C):
    """
    IAF time encoding parameter check.

    Determine whether a signal encoded with an Integrate-and-Fire
    neuron with the specified parameters can be perfectly recovered.

    Parameters
    ----------
    u : array_like of floats
        Signal to test.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Decoder bias.
    d : float
        Decoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.

    Returns
    -------
    rec : bool
        True if the specified signal is recoverable.

    Raises
    ------
    ValueError
        When the signal cannot be perfectly recovered.
    """

    c = np.max(np.abs(u))
    factor = 1
    if c >= b:
        raise ValueError('bias too low')
    r = factor * C * d / (b - c) * bw / np.pi

    if not np.isreal(r):
        raise ValueError('reconstruction condition not satisfied')
    elif r >= 1:
        raise ValueError('reconstruction condition not satisfied;' +
                         'try raising b or reducing d')
    else:
        return True


def iaf_encode(u, dt, b, d, R=np.inf, C=1.0, dte=0, y=0.0, interval=0.0,
               quad_method='trapz', full_output=False):
    """
    IAF time encoding machine.

    Encode a finite length signal with an Integrate-and-Fire neuron.

    Parameters
    ----------
    u : array_like of floats
        Signal to encode.
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.
    dte : float
        Sampling resolution assumed by the encoder (s).
        This may not exceed `dt`.
    y : float
        Initial value of integrator.
    interval : float
        Time since last spike (in s).
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal) when the
        neuron is ideal; exponential Euler integration is used
        when the neuron is leaky.
    full_output : bool
        If set, the function returns the encoded data block followed
        by the given parameters (with updated values for `y` and `interval`).
        This is useful when the function is called repeatedly to
        encode a long signal.

    Returns
    -------
    s : ndarray of floats
        If `full_output` == False, returns the signal encoded as an
        array of time intervals between spikes.
    [s, dt, b, d, R, C, dte, y, interval, quad_method, full_output] : list
        If `full_output` == True, returns the encoded signal
        followed by updated encoder parameters.

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in `u`.
    """

    Nu = len(u)
    if Nu == 0:
        if full_output:
            return np.array((), np.float), dt, b, d, R, C, dte, y, interval, \
                   quad_method, full_output
        else:
            return np.array((), np.float)

    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:
        # Resample signal and adjust signal length accordingly:
        M = int(dt / dte)
        u = scipy.signal.resample(u, len(u) * M)
        Nu *= M
        dt = dte

    # Use a list rather than an array to save the spike intervals
    # because the number of spikes is not fixed:
    s = []

    # Choose integration method:
    if np.isinf(R):
        if quad_method == 'rect':
            compute_y = lambda y, i: y + dt * (b + u[i]) / C
            last = Nu
        elif quad_method == 'trapz':
            compute_y = lambda y, i: y + dt * (b + (u[i] + u[i + 1]) / 2.0) / C
            last = Nu - 1
        else:
            raise ValueError('unrecognized quadrature method')
    else:

        # When the neuron is leaky, use the exponential Euler method to perform
        # the encoding:
        RC = R * C
        compute_y = lambda y, i: y * np.exp(-dt / RC) + R * (1 - np.exp(-dt / RC)) * (b + u[i])
        last = Nu

    # The interval between spikes is saved between iterations rather than the
    # absolute time so as to avoid overflow problems for very long signals:
    for i in range(last):
        y = compute_y(y, i)
        interval += dt
        if y >= d:
            s.append(interval)
            interval = 0.0
            y -= d

    if full_output:
        return [np.array(s), dt, b, d, R, C, dte, y, interval, \
                quad_method, full_output]
    else:
        return np.array(s)
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def iaf_decode(s,dur, dt, bw, b, d, R=np.inf, C=1.0):
    """
    IAF time decoding machine.

    Decode a finite length signal encoded with an Integrate-and-Fire
    neuron.

    Parameters
    ----------
    s : ndarray of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    R : float
        Neuron resistance.
    C : float
        Neuron capacitance.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.
    """

    Ns = len(s)
    print(Ns)
    if Ns < 2:
        raise ValueError('s must contain at least 2 elements')

    # Cast s to an ndarray to permit ndarray operations:
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1] + ts[1:]) / 2
    Nsh = len(tsh)

    t = np.arange(0, dur, dt)

    bwpi = bw / np.pi
    RC = R * C

    # Compute G matrix and quanta:
    G = np.empty((Nsh, Nsh), np.complex)
    if np.isinf(R):
        for j in range(Nsh):
            temp = scipy.special.sici(bw * (ts - tsh[j]))[0] / np.pi
            for i in range(Nsh):
                G[i, j] = temp[i + 1] - temp[i]
        q = C * d - b * s[1:]
    else:
        for i in range(Nsh):
            for j in range(Nsh):

                # The code below is functionally equivalent to (but
                # considerably faster than) the integration below:
                #
                # f = lambda t:np.sinc(bwpi*(t-tsh[j]))*bwpi*np.exp((ts[i+1]-t)/-RC)
                # G[i,j] = scipy.integrate.quad(f, ts[i], ts[i+1])[0]
                if ts[i] < tsh[j] and tsh[j] < ts[i + 1]:
                    G[i, j] = (-1j / 4) * np.exp((tsh[j] - ts[i + 1]) / RC) * \
                              (2 * se.ei((1 - 1j * RC * bw) * (ts[i] - tsh[j]) / RC) -
                               2 * se.ei((1 - 1j * RC * bw) * (ts[i + 1] - tsh[j]) / RC) -
                               2 * se.ei((1 + 1j * RC * bw) * (ts[i] - tsh[j]) / RC) +
                               2 * se.ei((1 + 1j * RC * bw) * (ts[i + 1] - tsh[j]) / RC) +
                               np.log(-1 - 1j * RC * bw) + np.log(1 - 1j * RC * bw) -
                               np.log(-1 + 1j * RC * bw) - np.log(1 + 1j * RC * bw) +
                               np.log(-1j / (-1j + RC * bw)) - np.log(1j / (-1j + RC * bw)) +
                               np.log(-1j / (1j + RC * bw)) - np.log(1j / (1j + RC * bw))) / np.pi
                else:
                    G[i, j] = (-1j / 2) * np.exp((tsh[j] - ts[i + 1]) / RC) * \
                              (se.ei((1 - 1j * RC * bw) * (ts[i] - tsh[j]) / RC) -
                               se.ei((1 - 1j * RC * bw) * (ts[i + 1] - tsh[j]) / RC) -
                               se.ei((1 + 1j * RC * bw) * (ts[i] - tsh[j]) / RC) +
                               se.ei((1 + 1j * RC * bw) * (ts[i + 1] - tsh[j]) / RC)) / np.pi

        q = C * (d + b * R * (np.exp(-s[1:] / RC) - 1))

    # Compute the reconstruction coefficients:
    c = np.dot(np.linalg.pinv(G, __pinv_rcond__), q)

    # Reconstruct signal by adding up the weighted sinc functions.
    u_rec = np.zeros(len(t), np.complex)
    for i in range(Nsh):
        u_rec += np.sinc(bwpi * (t - tsh[i])) * bwpi * c[i]
    return np.real(smooth(u_rec,100))
