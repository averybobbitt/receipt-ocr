import argparse
import os

from utils import perform_ocr


def main():
    # define and parse cli arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to input image")
    args = parser.parse_args()

    # check if image with given path exists
    if not os.path.exists(args.image):
        raise FileNotFoundError("The given image does not exist.")

    # do the magic here
    text = perform_ocr(args.image)

    # show the raw output of the OCR process
    print(text)


if __name__ == "__main__":
    main()
