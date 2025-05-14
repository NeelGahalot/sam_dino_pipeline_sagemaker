from PIL import Image, ExifTags

def print_exif_tags(image_path):
    try:
        with Image.open(image_path) as img:
            exif = img._getexif()

            if exif is not None:
                print("EXIF keys and their tags:")
                for key, value in exif.items():
                    tag = ExifTags.TAGS.get(key, f"Unknown ({key})")
                    print(f"{key} ({tag}): {value}")
            else:
                print("No EXIF data found.")
    except Exception as e:
        print(f"Error reading EXIF data: {e}")

print_exif_tags('/teamspace/studios/this_studio/florence_docker/assembli_ai.png')
