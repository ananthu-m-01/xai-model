from pathlib import Path
import os

def print_directory_structure(path, level=0):
    path_obj = Path(path)
    spaces = '    ' * level
    print(f'{spaces}{path_obj.name}/')
    
    for item in path_obj.iterdir():
        if item.is_dir():
            print_directory_structure(item, level + 1)
        else:
            print(f'{spaces}    {item.name}')

if __name__ == "__main__":
    base_path = Path(__file__).parent
    print("Dataset structure:")
    print_directory_structure(base_path)