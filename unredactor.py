import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import glob
import io
import sys
import spacy

# Load spaCy's high-performance transformer-based model
nlp = spacy.load("en_core_web_trf")

def get_entity(text):
    """
    Prints the entity inside of the text, displaying all recognized names.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Directly print all names labeled as PERSON
            print("Person Entity:", ent.text)

def doextraction(glob_text):
    """
    Get all the files from the given glob and pass them to the extractor.
    """
    for thefile in glob.glob(glob_text):
        with io.open(thefile, 'r', encoding='utf-8') as f:
            text = f.read()
            print(f"\nProcessing file: {thefile}")
            get_entity(text)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        doextraction(sys.argv[1])
    else:
        print("Please provide a file pattern to process, e.g., 'train/pos/*.txt'")

#  python unredactor.py "C:\Users\srees\Downloads\aclImdb\train\pos\12260_10.txt"