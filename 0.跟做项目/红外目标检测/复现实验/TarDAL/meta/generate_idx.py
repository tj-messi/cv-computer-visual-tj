# Let's generate the text file with the required format.
import os

def generate_pred():
    # Define the file path
    file_path = '/CV/zzx/TarDAL-main/data/m3fd/meta/pred.txt'

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Generate the lines and write to the file
        for i in range(4200):
            file.write(f"{i:05d}.png\n")


def get_train_val():
    # Specify the directory you want to check
    directory = '/CV/zzx/zjz-dataset/M3FD-yolo/image/val'  # Replace with your folder path

    # Define the output text file path
    output_file = '/CV/zzx/TarDAL-main/data/m3fd/meta/val.txt'

    # Open the file to write the filenames
    with open(output_file, 'w') as file:
        # Loop through all files in the folder
        for filename in os.listdir(directory):
            # Check if it's a file (not a directory)
            if os.path.isfile(os.path.join(directory, filename)):
                file.write(f"{filename}\n")

    print(f"Filenames have been written to {output_file}")

if __name__ == '__main__':
    get_train_val()

