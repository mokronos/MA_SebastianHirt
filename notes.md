# Notes

Just some notes that don't belong in the README. 

# Dataset

https://eezkni.github.io/publications/ESIM.html

## MOS: Mean Opinion Score

Often between 1 and 5, but can be any number. In this case it seems like its between 0 and 100, or its normalized after.

## PSNR: Peak Signal to Noise Ratio

Used to measure the quality of reconstruction of lossy compression algorithms.
PSNR measures the ratio of the original image to the error (noise) in the reconstructed image.
So with the same image, a higher PSNR means that there is a lower error.

## Goal

Should the goal be to minimize the error or be as close as possible to the MOS?
An algorithm that recognizes text even if the human can't would be useless.
If we would use that as a metric for compression algorithms, the results might be images that humans can't read but computers can.
So the objective metric should be as close as possible to the MOS.
If we plot MOS vs. Metric we want to have a straight line.
