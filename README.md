# Master Thesis (Sebastian Hirt)
![Unit Tests](https://github.com/Mokronos/MA_SebastianHirt/workflows/Unit%20Tests/badge.svg)

# Setup

Just run
```bash
$ pip install -r requirements.txt
```
to install all the dependencies. Best in a virtual environment with Python 3.8.10.

# Project Structure

```
.
├── codecs                  # Codecs used for distorting images, read README.md in /codecs
├── data                    # Data for the project
│   ├── raw                 # Raw data, not used in the project
│   └── gt                  # Annotated ground truth data
├── exp                     # some experimental files, not used in the final thesis
├── images                  # Generated images, non-graph images
│   ├── analysis            # all of the graphs
│   ├── external            # downloaded images, e.g. from papers, own images
├── latex                   # Latex source code for generating the thesis
├── literature              # Some important papers, but most are in the thesis/bibfile
├── organizing              # Initial task pdf
├── results                 # Predictions of OCR and result tables
│   ├── pred                # Predictions of OCR in txt and csv
│   └── summaries           # Result tables, intermediate DataFrames
├── src                     # Code, see README.md in src/
├── test                    # Unit tests
├── README.md               # This file
├── notes.md                # notes file used along the way for planning
└── requirements.txt        # Python dependencies
```

# Data

Download data [here](https://eezkni.github.io/publications/ESIM.html#:~:text=You%20can%20download%20the%20SCID%20as%20well%20as%20the%20supporting%20file%20via%20the%20OneDrive%3A%20Download%20SCID) and place into data folder with MOS_SCID.txt file (data/raw/scid/).

Data distorted by HEVC/VVC codecs is copied there too. Read codec [README](codecs/README.md) for more information.

# Testing

Run

```bash  
$ python -m unittest
```
in project root, to run all tests.

I think i wrote 1 test for converting the MATLAB nonlinear transformation code to python. This will fail now as the transformation code was adjusted.
And one for character error rate :).
So, broad test coverage was just an early ambition.

# Questions

Feel free to contact me at sebastian.hirt@fau.de and I'll try my best to answer any questions.

# Timeline

- Begin: 01.02.2023
- End: 31.07.2023
