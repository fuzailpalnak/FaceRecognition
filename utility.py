import os


def make_directory(current_dir, folder_name):
    new_dir = os.path.join(current_dir, folder_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir