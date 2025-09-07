# Ideal, Natural, & Flat-top -Sampling
Write a simple Python program for the construction and reconstruction of ideal, natural, and flattop sampling.
# Aim
To study and observe the construction and reconstruction of signals using Ideal (Impulse), Natural, and Flat-top Sampling techniques.
# Tools required
Computer with Python IDE / Google Colab
Libraries: NumPy, Matplotlib, SciPy
# Program
```
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, lfilter

# -------------------------------
# Helper: Low-pass Filter
# -------------------------------
def lowpass_filter(x, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, x)

# -------------------------------
# IMPULSE SAMPLING
# -------------------------------
fs1 = 100
t1 = np.arange(0, 1, 1/fs1)
f = 5
cont_imp = np.sin(2*np.pi*f*t1)
samp_imp = cont_imp.copy()                       # ideal impulse samples
rec_imp  = resample(samp_imp, len(t1))           # simple resample for demo

# -------------------------------
# NATURAL SAMPLING
# -------------------------------
fs2 = 1000
t2 = np.arange(0, 1, 1/fs2)
fm = 5
msg_nat = np.sin(2*np.pi*fm*t2)

pulse_rate = 50
pulse_train = np.zeros_like(t2)
pulse_width = int(fs2/pulse_rate/2)              # 50% duty
for i in range(0, len(t2), int(fs2/pulse_rate)):
    pulse_train[i:i+pulse_width] = 1

samp_nat = msg_nat * pulse_train                 # natural sampled
rec_nat  = lowpass_filter(samp_nat, 10, fs2)     # LPF to reconstruct

# -------------------------------
# FLAT-TOP SAMPLING
# -------------------------------
fs3 = 1000
t3 = np.arange(0, 1, 1/fs3)
msg_flat = np.sin(2*np.pi*fm*t3)

pulse_idx = np.arange(0, len(t3), int(fs3/pulse_rate))
pulse_width_samples = int(fs3/(2*pulse_rate))    # 50% duty
samp_flat = np.zeros_like(t3)

for idx in pulse_idx:
    val = msg_flat[idx]
    end_idx = min(idx + pulse_width_samples, len(t3))  # <-- bounds safe
    samp_flat[idx:end_idx] = val

rec_flat = lowpass_filter(samp_flat, 2*fm, fs3)

# -------------------------------
# PLOTS (3x3 Grid): rows = C/S/R, cols = Impulse/Natural/Flat-top
# -------------------------------
plt.figure(figsize=(15,10))

# Impulse (Column 1)
plt.subplot(3,3,1); plt.plot(t1, cont_imp);                 plt.title("Impulse - Continuous");    plt.grid(True)
plt.subplot(3,3,4); plt.stem(t1, samp_imp);                 plt.title("Impulse - Sampled");       plt.grid(True)
plt.subplot(3,3,7); plt.plot(t1, rec_imp, 'g');             plt.title("Impulse - Reconstructed");  plt.grid(True)

# Natural (Column 2)
plt.subplot(3,3,2); plt.plot(t2, msg_nat);                  plt.title("Natural - Continuous");    plt.grid(True)
plt.subplot(3,3,5); plt.plot(t2, samp_nat);                 plt.title("Natural - Sampled");       plt.grid(True)
plt.subplot(3,3,8); plt.plot(t2, rec_nat, 'g');             plt.title("Natural - Reconstructed");  plt.grid(True)

# Flat-top (Column 3)
plt.subplot(3,3,3); plt.plot(t3, msg_flat);                 plt.title("Flat-top - Continuous");   plt.grid(True)
plt.subplot(3,3,6); plt.plot(t3, samp_flat);                plt.title("Flat-top - Sampled");      plt.grid(True)
plt.subplot(3,3,9); plt.plot(t3, rec_flat, 'g');            plt.title("Flat-top - Reconstructed"); plt.grid(True)

plt.tight_layout()
plt.show()

```
# Output Waveform
<img width="1498" height="989" alt="image" src="https://github.com/user-attachments/assets/4f3cac04-545b-470e-9bc5-f7ea6a26b241" />

# Results
In Impulse sampling, the signal is sampled as impulses and can be perfectly reconstructed in theory.
In Natural sampling, the signal follows the input shape during sampling and reconstruction is close to the original.
In Flat-top sampling, each sample is held flat, causing slight distortion in reconstruction.
