#!/usr/bin/env python

"""Show a text-mode spectrogram using live microphone data."""
import argparse
import math
import numpy as np
import shutil
import pigpio
from collections import deque
import scipy.stats as stats

usage_line = ' press <enter> to quit, +<enter> or -<enter> to change scaling '

pi = pigpio.pi()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

try:
    columns, _ = shutil.get_terminal_size()
except AttributeError:
    columns = 80

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-l', '--list-devices', action='store_true',
                    help='list audio devices and exit')
parser.add_argument('-b', '--block-duration', type=float,
                    metavar='DURATION', default=25,
                    help='block size (default %(default)s milliseconds)')
parser.add_argument('-c', '--color', type=int, nargs=3,
                    metavar=('RED','GREEN','BLUE'), default=[255, 0, 0],
                    help='color of the LEDs (default %(default)s)')
parser.add_argument('-d', '--device', type=int_or_str,
                    help='input device (numeric ID or substring)')
parser.add_argument('-g', '--gain', type=float, default=10,
                    help='initial gain factor (default %(default)s)')
parser.add_argument('-r', '--range', type=float, nargs=2,
                    metavar=('LOW', 'HIGH'), default=[0, 200],
                    help='frequency range (default %(default)s Hz)')
parser.add_argument('-f', '--fade', action='store_true', default=False,
                    help='include to set the lights to fade through colors')
args = parser.parse_args()

low, high = args.range
if high <= low:
    parser.error('HIGH must be greater than LOW')

# The Pins. Use Broadcom numbers.
RED_PIN   = 16
GREEN_PIN = 20
BLUE_PIN  = 21

HEX_MAX = 16777215

if (args.fade):
    r, g, b = 255, 0, 0
else:
    r, g, b = args.color

def setLights(pin, brightness):
    if brightness>255:
        return
    pi.set_PWM_dutycycle(pin, brightness)

def reset_lights():
    setLights(RED_PIN, 0)
    setLights(GREEN_PIN, 0)
    setLights(BLUE_PIN, 0)

def bounded(x, l, h):
    return max(l, min(h, x))

def normalize(val, prev_vals, margin=0):
    val_max = np.percentile(prev_vals, 100-margin)
    val_min = np.percentile(prev_vals, margin*2)
    val_normed = (val-val_min)/max(1e-9,(val_max-val_min))
    return bounded(val_normed, 0, 1)

def curved_norm(val, prev_vals, margin=0):
    norm = normalize(val, prev_vals, margin)
    return bounded((norm+0.1)**6, 0, 1)
    #return (((2*(norm-0.5))**3)/2)+0.5


def int_to_hex(i):
    return '{0:06x}'.format(i)

def hex_to_rgb(h):
    return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))

def norm_to_rgb(norm):
    i = int(norm*HEX_MAX)
    h = int_to_hex(i)
    return hex_to_rgb(h)

def updateColor(color, step):
    color += step
    
    if color > 255:
        return 255
    if color < 0:
        return 0
        
    return color

def fadeStep(steps):
    global r, g, b

    if r == 255 and b == 0 and g < 255:
        g = updateColor(g, steps)
    
    elif g == 255 and b == 0 and r > 0:
        r = updateColor(r, -steps)
    
    elif r == 0 and g == 255 and b < 255:
        b = updateColor(b, steps)
    
    elif r == 0 and b == 255 and g > 0:
        g = updateColor(g, -steps)
    
    elif g == 0 and b == 255 and r < 255:
        r = updateColor(r, steps)
    
    elif r == 255 and g == 0 and b > 0:
        b = updateColor(b, -steps)

try:
    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)

    samplerate = sd.query_devices(args.device, 'input')['default_samplerate']
    print('Sampling at %s Hz' % samplerate)

    amp_max = 1e-9
    # note that the first number after maxlen is the number of milliseconds that will be preserved
    # for normalization


    norm_duration = 10000
    delay = 300

    prev_amps = deque([1e-9],maxlen=int(norm_duration/args.block_duration))
    norm_amps = deque([], maxlen=max(1,int(delay/args.block_duration)))
    noise_thresh = 1e9

    def callback(indata, frames, time, status):
        if (args.fade):
            fadeStep(1)
        if status:
            text = ' ' + str(status) + ' '
            print('\x1b[34;40m', text.center(columns, '#'),
                  '\x1b[0m', sep='')
        if any(indata):
            data = indata[:, 0]
            data = data * np.hanning(len(data))
            spectrum = np.abs(np.fft.rfft(data))
            freq = np.fft.fftfreq(len(data), 1/samplerate)

            spectrum = spectrum[:int(len(spectrum)/2)]
            freq = freq[:int(len(freq)/2)]
            band = spectrum[np.where(np.logical_and(freq>=low, freq<high))]
            freqPeak = freq[np.where(spectrum==np.max(spectrum))]

            amp = np.sum(band)
            global noise_thresh
            noise_thresh = min(amp, noise_thresh)
            amp=max(1e-9, amp-noise_thresh)

            prev_amps.append(amp)
            norm_amps.append(curved_norm(amp, prev_amps, 10))

            if (np.std(prev_amps)<1):
                norm_amps.clear()
                norm_amps.append(0)

            bright = np.average(norm_amps, weights=np.arange(1,len(norm_amps)+1))
            setLights(RED_PIN, r*bright)
            setLights(GREEN_PIN, g*bright)
            setLights(BLUE_PIN, b*bright)

            #print("peak frequency: %5d Hz  Brightness: %f"%(freqPeak,bright))
            print(int(amp/10)*'-')
        else:
            print('no input')

    with sd.InputStream(device=args.device, channels=1, callback=callback,
                        blocksize=int(samplerate * args.block_duration / 1000),
                        samplerate=samplerate):
        print("Starting the lightshow...")
        while True:
            response = input()
            if response in ('', 'q', 'Q'):
                break
            for ch in response:
                if ch == '+':
                    args.gain *= 2
                elif ch == '-':
                    args.gain /= 2
                else:
                    print('\x1b[31;40m', usage_line.center(columns, '#'),
                          '\x1b[0m', sep='')
                    break

    reset_lights()
except KeyboardInterrupt:
    parser.exit('Interrupted by user')
    reset_lights()
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
