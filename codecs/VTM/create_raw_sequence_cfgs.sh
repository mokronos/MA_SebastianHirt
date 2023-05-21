#!/bin/bash
for filename in Image/*.ppm; do
    fbname=$(basename "$filename" .ppm)
    filenameraw="./"$fbname".raw"
    config=cfgs_seq/$fbname".cfg"
    width=$(identify -format '%w' "$filename")
    height=$(identify -format '%h' "$filename")
    echo "#======== File I/O ===============" > "$config"
    echo "InputFile                     : $filenameraw" >> "$config"
    echo "InputBitDepth                 : 8           # Input bitdepth" >> "$config"
    echo "InputChromaFormat             : 444         # Ratio of luminance to chrominance samples" >> "$config"
    echo "FrameRate                     : 60          # Frame Rate per second" >> "$config"
    echo "FrameSkip                     : 0           # Number of frames to be skipped in input" >> "$config"
    echo "SourceWidth                   : $width      # Input  frame width" >> "$config"
    echo "SourceHeight                  : $height     # Input  frame height" >> "$config"
    echo "FramesToBeEncoded             : 1           # Number of frames to be coded" >> "$config"
    echo "InputColourSpaceConvert       : RGBtoGBR    # Non-normative colour space conversion to apply to input video" >> "$config"
    echo "SNRInternalColourSpace        : 1           # Evaluate SNRs in GBR order" >> "$config"
    echo "OutputInternalColourSpace     : 0           # Convert recon output back to RGB order. Use --OutputColourSpaceConvert GBRtoRGB on decoder to produce a matching output file." >> "$config"
    echo " " >> "$config"
    echo "Level                         : 6.2" >> "$config"
done
