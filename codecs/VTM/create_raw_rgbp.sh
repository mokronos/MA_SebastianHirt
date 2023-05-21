#!/bin/bash
for filename in Image/*.ppm; do
    convert "$filename" -interlace plane rgb:./$(basename "$filename" .ppm).raw
done
