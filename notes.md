# Notes

Just some notes that don't belong in the README. 

# Datasets

## SCID
https://eezkni.github.io/publications/ESIM.html
paper: https://eezkni.github.io/publications/journal/ESIM/ESIM_ZKNI_TIP17.pdf
## CIQAD
https://sites.google.com/site/ciqadatabase/
paper: https://sci-hub.ru/10.1016/j.jvcir.2014.11.001
## SIQAD
https://sites.google.com/site/subjectiveqa/
paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7180347


## Metrics

### MOS: Mean Opinion Score

Often between 1 and 5, but can be any number. In this case it seems like its between 0 and 100, or its normalized after.

### PSNR: Peak Signal to Noise Ratio

Used to measure the quality of reconstruction of lossy compression algorithms.
PSNR measures the ratio of the original image to the error (noise) in the reconstructed image.
So with the same image, a higher PSNR means that there is a lower error.

### CER: Character Error Rate 
Compares the total amount of characters to the minimum amount of insertions, deletions and substitutions needed to transform the recognized text into the ground truth text.

CES = (insertions + deletions + substitutions) / total characters

### Goal

Should the goal be to minimize the error or be as close as possible to the MOS?
An algorithm that recognizes text even if the human can't would be useless.
If we would use that as a metric for compression algorithms, the results might be images that humans can't read but computers can.
So the objective metric should be as close as possible to the MOS.
If we plot MOS vs. Metric we want to have a straight line.


## Text Recognition Algorithms

### Tesseract

https://github.com/tesseract-ocr/tesseract

paper: https://ieeexplore.ieee.org/document/4376991
but author only author until 2018

settings: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html

### Paddle OCR

https://github.com/PaddlePaddle/PaddleOCR
paper: https://arxiv.org/abs/2009.09941
python quickstart: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/quickstart_en.md
model summary: https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_en/algorithm_overview_en.md
not sure yet how to use specific models

### Ocropus

https://github.com/ocropus/ocropy
no paper
seems to be experimental, not as stable

### Kraken OCR

https://github.com/mittagessen/kraken
paper: https://arxiv.org/ftp/arxiv/papers/1703/1703.09550.pdf

### Microsoft OCR

https://learn.microsoft.com/en-us/azure/cognitive-services/computer-vision/quickstarts-sdk/client-library?tabs=visual-studio&pivots=programming-language-python

### MMOCR

https://github.com/open-mmlab/mmocr

### Vedastr
https://github.com/Media-Smart/vedastr
small? but simple library

## Models

Maybe look for stuff here: https://huggingface.co/models?sort=downloads&search=ocr

### Mask OCR
https://paperswithcode.com/paper/maskocr-text-recognition-with-masked-encoder


## Datasets

### Applicable

- Paper http://smartviplab.org/pubilcations/SCID/zkni_TIP_ESIM_2017.pdf
    - Alternative Paper: https://ieeexplore.ieee.org/document/8266580
    - Dataset Download/Website http://smartviplab.org/pubilcations/SCID.html

- Paper https://arxiv.org/pdf/2008.08561.pdf
    - link is dead, contacted all 3 authors, no response
### Not applicable

- https://sites.google.com/site/zhangxinf07/fg-iqa (FG-IQA)
    - not applicable, not screen content, nor text of any kind

- https://live.ece.utexas.edu/research/Quality/live_multidistortedimage.html
    - no text

### Not sure
- https://iopscience.iop.org/article/10.1088/1742-6596/1828/1/012033
    - http://cvbrain.cn/download/#
    - document image quality assessment
    - no reference images, that's fine
    - just document images, not screen content with MOS scores
- https://www.sciencedirect.com/science/article/abs/pii/S1296207417303382
    - no link to dataset found
    - might be applicable

## Annotations

Image annotation formats: [link](https://www.edge-ai-vision.com/2022/04/exploring-data-labeling-and-the-6-different-types-of-image-annotation/)

- Bounding boxes
    - Recognized text plus bounding box coordinates
    - Threshold margin separating different text elements
    - One bounding box for each element
    - left, top, right, bottom
    - either text elements or single characters, flexible
    - sometimes even curved bounding boxes
- Text
    - Recognized text
    - Confidence

Annotations from algorithms are similar, bbox + text. So it would be reasonable to adjust to the dataset.

## Other Literature

- https://paddleocr.bj.bcebos.com/ebook/Dive_into_OCR.pdf
    - OCR book

- https://engineering.fb.com/2018/09/11/ai-research/rosetta-understanding-text-in-images-and-videos-with-machine-learning/
    - Facebook paper on text recognition
    - rosetta

- https://research.aimultiple.com/ocr-technology/
    - current state of OCR

- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=156468
    - Historical Review of OCR Research and Development 
    - way too old, 1992

- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9151144
    - Handwritten Optical Character Recognition (OCR): A Comprehensive Systematic Literature Review (SLR)
    - handwritten, but maybe still interesting, as there aren't many algorithms that are trained on screen content
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9183326
    - Text extraction using OCR: A Systematic Review
    - might be a good overview

## schedule

| Month | Tasks |
| --- | --- |
| 1 | Choose a specific research question. Conduct a thorough literature review to understand the state-of-the-art in text recognition algorithms and MOS on compressed screen content data. |
| 2 | Collect and preprocess your dataset. Implement and compare different text recognition algorithms on the dataset.|
| 3 | Analyze the collected data and compare the performance of the text recognition algorithms against MOS. Draft the introduction and methodology chapters.|
| 4 | Evaluate the robustness and limitations of the text recognition algorithms and MOS data. Discuss the results and draw conclusions on the research question. Write Evaluation, identify what else is needed to explain results. |
| 5 | Refine the thesis chapters, including the discussion and conclusion. Draft the abstract and connect chapters. |
| 6 | Finalize the thesis and prepare for submission. Review and proofread the thesis. |

## Random ideas
- Subjective metric should be comparison with original image, not absolute. Double stimulus.(for SCID dataset)
- Why not train a model to predict human score?

## Notes

- [x] add bounding boxes in ground truth, as detection is mentioned too in the task
    - use bounding boxes to run OCR on each box
    - plan
        - just sort every word by y coordinate, then by x coordinate
        - easyocr and tesseract have bbox data and corresponding words
        - combine into long string and compare with ground truth
        - need to modify ground truth
        - need to check exact positions of words
        - might make sense to go straight to bounding box labeling
        - doesn't work, bounding boxes have different heights for bigger letters, gets messy
        - plus, bounding box definitions/thresholds are different for tesseract and easyocr
    - or use dameru levenshtein distance
        - would work, with "range" set to word
    - using full data representations, but just take text in the order given by the algorithm. Its the same order the direct string methods have, but I'm carrying more information.

- [ ] write section for TER/CER and MOS
    - need to flesh out and add sources
    - add other common metrics (mentioned in the task as well)

- [ ] check which for what images/compressions the correlation is high
    - need to color code the figures to differentiate between the different images, compressions and qualities
    - need to custom colorcode probably, or check seaborn library, works well with pandas

- [ ] figure out the inner workings of tesseract
    - when it fails, why does it fail?
    - what preprocessing is applied automatically?
    - still no result for some compressed images

- [ ] document quality assessment dataset; missing MOS? Check if in last package
    - scores are in last package, but still need to download all
    - need to connect first then unzip, .zip.001 to .zip.024 extensions

- [ ] make config file with all the paths
    - generate all the paths so the folders exist
## Questions
