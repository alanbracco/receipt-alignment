# Receipt Alignment

It is a Python script that applies image processing to vertically align a receipt in the picture (by rotating the entire image). It can look for the receipt contour or analyse the text to achieve the result.

## Requirements

Requirements are listed in `requirements.txt` and `Pipfile`.
They are:
- opencv-python (v4.5.1.48)
- scipy (v1.6.2)
- pytesseract (v0.3.7)

It also requires the *tesseract-ocr* engine installed in the system. In Ubuntu it can be installed by executing `sudo apt-get install tesseract-ocr=4.1.1-2build2`. For Windows, links to packages are found [here](https://tesseract-ocr.github.io/tessdoc/Downloads.html). Particularly this script was tested using version 3.02 from SourceForge.

## How to use it?

### Command line interface

`python AlignReceipt.py --input {path_to_input_file} --output {path_to_output_file}`

Both arguments (input and output) are required. Run `python AlignReceipt.py --help` for more information.

### Running from python

The script provides a `main` method than can be called from another Python script (if AlignReceipt is accesible from there). It does not necessarily writes the result to disk (unless output_path is given)

```
import AlignReceipt

input_path = 'path/to/input/file'

# Get the aligned image
aligned_image = AlignReceipt.main(input_path)

output_path = 'output/file/path'
# Get the aligned image and write to disk
aligned_image = AlignReceipt.main(input_path, output_path)
```

## Image examples

Examples are found in *ImageExamples* folder. Some of the examples are variatons of the The ExpressExpense SRD (sample receipt dataset) samples.

