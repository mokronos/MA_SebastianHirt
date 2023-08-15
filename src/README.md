# Code

This should give an overview of the code structure.

## Files

```
.
├── compare_codecs.py       # compare ocr prediction on codec distorted images with ground truth and reference prediction + calculate certain stats
├── compare_dist.py         # compare ocr prediction on distorted images with ground truth and reference prediction + calculate certain stats
├── compare_ref.py          # calculate cer for OCR on reference images compared to GT
├── config.py               # define what images/distortions/qualities to use for each experiment, define paths so that they are consistenstent across the project for saving and loading data
├── create_gt.py            # initial script to create ground truth, later abandoned to not bias towards one OCR method
├── exp.py                  # some small experimental scripts, not used in the final thesis
├── ezocr.py                # EasyOCR test script, not used in the final thesis
├── helpers.py              # helper functions for loading images, calculating CER, etc.
├── plotting.py             # plotting functions for generating graphs
├── predict.py              # predict OCR on images, save to csv and txt
└── tables.py               # generate result tables
```


## Main Pipeline

1. create ground truth manually and save to data/gt/
2. run predict.py to predict OCR on codec/distorted/reference images and save to results/pred/
3. run corresponding compare_*.py script to compare predictions with ground truth and reference prediction and save to results/summaries/
4. run plotting.py/tables.py to generate graphs/tables and save to images/analysis/ or results/summaries

## Philosophy

All the different steps are saved to disk and loaded in the next step.
This makes checking for bugs in the intermediate results easier and saves time for steps that take a decently long time (OCR prediction on 1800 images).
To help with saving/loading the config file defines the paths, so that if something changes, only the config file needs to change.

The data is generally saved in pandas DataFrames.
The DataFrame is set up with all the necessary combinations of images necessary in a setup function in the compare scripts.
So for instance, the compare_dist DataFrame has (if all images were to be used) 1800 rows, but is then expanded with an extra column for the OCR algorithm used (EasyOCR or Tesseract), which makes it 3600 rows.
The same thing is done for the target of the CER calculation, which is either the ground truth or the prediction on the reference image.
After, the CER and other metrics are added by looping over all rows and calculating the metric depending on the target/ocr algorithm and other things.
Those DataFrames are the saved to be loaded in the plotting.py/tables.py scripts to visualize the data.

A lambda function is generally used to apply the calculation of a new column to the DataFrame.
All files have some functions and execute those functions at the bottom, so that some functions can be easily commented out, if they are not needed.
For plotting there is a pipeline function, which executes all of the plotting functions used in the latex code for the thesis.

## OCR Methods

Both EasyOCR and pytesseract were installed with pip.
After, the default recommended models were installed automatically.
