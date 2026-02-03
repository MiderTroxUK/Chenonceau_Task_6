
import sys
import re

def extract_raw_strings(filename, min_len=4):
    with open(filename, 'rb') as f:
        data = f.read()
        # Find sequences of printable characters
        # specific to PDF text often being in () or just generally printable
        # This is a very rough approximation
        text = ""
        # Filter for printable chars
        chars = []
        for b in data:
            if 32 <= b <= 126 or b in (10, 13):
                chars.append(chr(b))
            else:
                if len(chars) >= min_len:
                    text += "".join(chars) + "\n"
                chars = []
        if len(chars) >= min_len:
            text += "".join(chars) + "\n"
    return text

files = [
    r"c:\Users\johnh\OneDrive - Cranfield University\Group4 Project\Task3\draftTask3.pdf",
    r"c:\Users\johnh\OneDrive - Cranfield University\Group4 Project\Task3\Pierre.B - Truss - SA.pdf"
]

for file_path in files:
    print(f"--- RAW STRINGS FROM {file_path} ---")
    content = extract_raw_strings(file_path)
    # Print only first 2000 chars to avoid massive output if it's junk
    print(content[:2000])
    print("--- END ---")
