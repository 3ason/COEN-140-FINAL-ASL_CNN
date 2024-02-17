from PIL import Image
from os import listdir, path, mkdir, makedirs
import sys

accepted_image_types = ['jpg', 'png']


def convert_and_store(root_path, new_root_path):
    try:
        dir_list = listdir(root_path)
    except OSError as e:
        if not path.exists(root_path):
            print("The directory as specified does not exist")
        print(e)

        dir_list = []

    if not path.exists(gray_path):
        try:
            makedirs(gray_path)
        except OSError as e:
            print("The new path as specified is not valid")
            print(e)
            print("Terminating Process")
            sys.exit()

    for each in dir_list:
        temp_path = path.join(root_path, each)

        # if the object is a file, we convert to grayscale and store
        if path.isfile(temp_path):
            if is_accepted_image_type(each):
                save_gray_photo(temp_path, new_root_path, each)
            else:
                print(each + "has an invalid file type, must be JPEG or PNG")

        elif path.isdir(temp_path):
            new_nested_path = path.join(new_root_path, temp_path.split(sep='/')[-1])

            if not path.exists(new_nested_path):
                mkdir(new_nested_path)

            for inner_path in listdir(temp_path):
                nested_path = path.join(temp_path, inner_path)
                
                if path.isfile(nested_path):
                    save_gray_photo(nested_path, new_nested_path, inner_path)

                elif path.isdir(nested_path):
                    convert_and_store(nested_path, path.join(new_root_path, inner_path))


def save_gray_photo(path_to_original, path_to_destination_folder, file_name):
    index = file_name.find('.')
    full_path = path.join(path_to_destination_folder, file_name[:index] + '_gray' + file_name[index:])

    img = Image.open(path_to_original).convert('L')
    img.save(full_path)


def is_accepted_image_type(file_name) -> bool:
    return file_name.split(sep='.')[1] in accepted_image_types


if __name__ == '__main__':
    # first arg is root path of image directory
    dir_root = sys.argv[1]
    assert type(dir_root) == str

    # second arg is desired path of new grayscale root directory
    gray_path = sys.argv[2]
    assert type(gray_path) == str

    convert_and_store(dir_root, gray_path)




