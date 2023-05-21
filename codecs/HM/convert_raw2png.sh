#!/bin/bash

q=37
# for q in {22..37}; do
for filename in Image/*.ppm; do
    fbname=$(basename "$filename" .ppm)

    # Decoding
    /home/och/ReadersAndCodecs/HM/bin/TAppDecoderStatic -b "$fbname"_CTC_"$q".bin -o "$fbname"_CTC_dec_reco_"$q".raw --OutputColourSpaceConvert=GBRtoRGB --OutputBitDepth=8

    # Reconstruction
    filenameraw=$fbname"_CTC_dec_reco_"$q".raw"
    filenamepng_dec=$fbname"_CTC_dec_reco_"$q".ppm"
    width=$(identify -format '%w' $filename)
    height=$(identify -format '%h' $filename)
    do_back_convertion="convert -size "$width"x"$height" -depth 8 -interlace plane rgb:$filenameraw $filenamepng_dec"

    echo $do_back_convertion
    $do_back_convertion
done
# done
