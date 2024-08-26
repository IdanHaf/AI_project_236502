import os

def count_objects_in_directory(directory_path):
    try:
        # Get the list of all files and directories in the specified directory
        objects = os.listdir(directory_path)
        # Count the number of objects
        num_objects = len(objects)
        print(f'There are {num_objects} objects in the directory: {directory_path}')
    except FileNotFoundError:
        print(f'The directory "{directory_path}" does not exist.')
    except Exception as e:
        print(f'An error occurred: {e}')

# Specify the directory path
directory_path = 'images'

# Call the function
count_objects_in_directory(directory_path)
