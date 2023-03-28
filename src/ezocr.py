import easyocr

folder_path = 'data/raw/ReferenceSCIs/'
# folder_path = 'data/raw/DistortedSCIs/'
reader = easyocr.Reader(['en'])

numbers = list(range(1, 41))
numbers = [x if x > 9 else f'0{x}' for x in numbers]

for number in numbers:
    # image_path = f'{folder_path}SCI01_1_5.bmp'
    image_path = f'{folder_path}SCI{number}.bmp'

    result = reader.readtext(image_path, detail=0, paragraph=True)
    print(f'OCRed SCI{number}, writing to file ...')
    # write to text file
    with open(f'labels/ezocr/SCI{number}_pred.txt', 'w') as f:
        for line in result:
            f.write(f"{line}\n")
