# How to download codecs and transform the images

- make the 3 scripts executable
- run ./install to clone and compile the codecs
- adjust the settings in ./transform as needed (q, codec)
- copy images in original folder and adjust extension in ./transform
``` bash
cp ../data/raw/scid/ReferenceSCIs/SCI(01|03|04|05|29).bmp images/scc/original/
cp ../data/raw/scid/ReferenceSCIs/SCI(01|03|04|05|29).bmp images/default/original/
```
- run ./transform with images as the arguments to transform the images
- copy from results/ to main data folder of project
``` bash
cp -r images/scc/results/* ../data/raw/scid/scc/
cp -r images/default/results/* ../data/raw/scid/default/
```
