# Latex Thesis Generation

This folder contains the source code for generating the thesis in PDF format.

## Requirements

```bash
sudo apt-get install texlive-latex-extra
```

I get the error that IEEEtran file is not found, so I also installed:

```bash
sudo apt-get install texlive-publishers
```

Then its fine.

## Usage

Just run `make` in this folder to generate the PDF, and all the log/aux files.

Run `make clean` to remove all generated files, but the PDF.

## Images

Images are mainly pulled directly from the sibling directories, so generating them first, might be necessary.
