#!/usr/bin/env python
import pyexr
import numpy as np
import sys
import os.path as op
from scipy.ndimage import rotate
from scipy.optimize import curve_fit
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
import numpy.fft as fft
import pandas as pd
from matplotlib.widgets import RectangleSelector
import json

# Computes MTF of ROI with slanted edge in EXR image.
# Usage: run python compute_mtf.py <image.exr>. Select ROI with slanted edge.
# Outputs: mtf.csv (SFR), mtf.json (parameters of GMMs that fit the MTF).

exposure = 0.02 # multiplicative factor for pixel values
x_resize_factor = 10 # factor for resizing image in x axis
lowest_amp = 0.5 # MTF amplitudes lower than this are clipped

# Event handler for rectangle selector
def onselect(eclick, erelease):
	x1, y1 = eclick.xdata, eclick.ydata
	x2, y2 = erelease.xdata, erelease.ydata
	global crop
	crop = np.array([[x1, y1], [x2, y2]]).astype(int)
	print(f"Rectangle selected: ({x1}, {y1}) to ({x2}, {y2})")

def gauss2(rho, a1, b1, c1, a2, b2, c2):
	term1 = a1 * np.exp(-((rho - b1) / c1) ** 2)
	term2 = a2 * np.exp(-((rho - b2) / c2) ** 2)
	return term1 + term2

def createFit(mtf_freq, mtf_amp):
	p0 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
	low_bounds = [-np.inf, -np.inf, 1e-6, -np.inf, -np.inf, 1e-6]
	up_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
	params, _ = curve_fit(gauss2, xdata=mtf_freq, ydata=mtf_amp, p0=p0, bounds=(low_bounds, up_bounds), method='trf')
	return params

def plotFreqVsAmp(mtf_freq, mtf_amp, lowest_amp, gParams, color='k'):
	plt.figure('MTF GM Fit')
	plt.scatter(mtf_freq, mtf_amp, label="MTF Amp vs MTF Freq", c=color, marker='.')
	plt.hlines(y=lowest_amp, xmin=0.0, xmax=0.5, linewidth=2, color=color, linestyle='--', alpha=0.5)
	plt.plot(mtf_freq, gauss2(mtf_freq, *gParams), label="GM Fit", c='b', linewidth=2, alpha=0.5)
	plt.ylim([0.0, 1.0])
	plt.xlabel('MTF Frequency')
	plt.ylabel('MTF Amplitude')
	plt.legend(loc='upper right')
	plt.grid(True)

def fit_curve_with_GMM(mtf_freq, mtf_amp, lowest_amp=0.5):
	freqs = mtf_freq
	amps  = mtf_amp
	amps = amps[freqs <= 0.5]
	last_valid_index = np.where(freqs <= 0.35)[0][-1]
	amps[last_valid_index:] = amps[last_valid_index]
	freqs = freqs[freqs <= 0.5]
	amps_clipped = np.clip(amps, lowest_amp, 1.0)
	gParams = createFit(freqs, amps_clipped)
	return gParams

if __name__ == "__main__":
	if len(sys.argv) == 1:
		sys.exit("specify name of image")

	# Read exr image with slanted edge
	fname_exr = sys.argv[1]
	ext = op.splitext(fname_exr)[-1]
	img = pyexr.read(fname_exr)

	# Select ROI with slanted edge
	fig, ax = plt.subplots()
	ax.imshow(np.clip(img*exposure, 0.0, 1.0))
	recSel = RectangleSelector(ax, onselect, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
	plt.show()

	# Crop ROI with slanted edge
	print("Crop:", crop)
	img = img[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0], :]
	pyexr.write(fname_exr.replace(ext,"_crop"+ext), img)

	# Show crop
	plt.imshow(np.clip(img*exposure, 0.0, 1.0))
	plt.title("Image ROI")
	plt.show()

	# Compute edge slope and rotate
	norm = np.linalg.norm(img, axis=-1) #compute norm of image
	dx, dy = np.gradient(norm) #compute gradients in x and y
	slope = dx.sum() / dy.sum() #compute slope
	angle = np.arctan(slope)*180.0/np.pi
	img_rot = rotate(img, angle, reshape=False, mode="nearest")

	# Show rotated crop (vertical edge)
	plt.imshow(np.clip(img_rot*exposure, 0.0, 1.0))
	plt.title("Rotated ROI")
	plt.show()

	# Plot ESF (Edge Spread Function)
	esf = img_rot.mean(axis=0)
	for irgb, rgb in enumerate("rgb"):
		plt.plot(range(len(esf)), esf[:,irgb], label="ESF "+rgb, c=rgb)
	plt.title("ESF (Edge Spread Function)")
	plt.yscale("log")
	plt.legend()
	plt.show()

	# Compute the LSF (Line Spread Function) from the ESF and save as CSV. Columns are MTF_freq, MTF_amp (R, G, B, Lum).
	df = pd.DataFrame()
	for ichan, esf_chan in enumerate([esf[:,0], esf[:,1], esf[:,2], np.linalg.norm(esf, axis=-1)]):
		lsf = np.gradient(esf_chan, axis=0, edge_order=1)
		mtf_freqs = fft.fftshift(fft.fftfreq(len(lsf)))
		mtf_amps = np.abs(fft.fftshift(fft.fft(lsf)))
		mtf_freqs = mtf_freqs[len(mtf_freqs)//2:]
		mtf_amps = mtf_amps[len(mtf_amps)//2:]
		df[ichan] = mtf_amps / mtf_amps[0]
	df.index = mtf_freqs
	df.index, df = df.index.round(decimals=3), df.round(decimals=3)
	df.to_csv("mtf.csv", header=False)
	print("Wrote mtf.csv")

	# Fit curves (R, G, B, Lum), save fits in JSON and Plot.
	gParams_dict = {}
	for irgbY, rgbY in enumerate(["R", "G", "B", "Y"]):
		mtf_freqs, mtf_amps = df.index.values, df.values
		gParams = fit_curve_with_GMM(mtf_freqs, mtf_amps[:,ichan], lowest_amp=lowest_amp)
		gParams_dict[rgbY] = gParams.tolist()
		print("Fitted GM Parameters " + rgbY + ":", gParams)
	with open("mtf.json", "w") as f:
		json.dump(gParams_dict, f)
		print("Wrote mtf.json")
	plotFreqVsAmp(mtf_freqs, mtf_amps[:,ichan], lowest_amp, gParams, color='k')
	plt.show()