import os
import zipfile

def zip_every_two_files(folder_path, output_folder):
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Sort the files to ensure consistent pairing
    files.sort()

    # Iterate over the files and create zip files for every two files
    for i in range(0, len(files), 2):
        file1 = files[i]
        if i + 1 < len(files):
            file2 = files[i + 1]

            # Create a zip file name based on the pair
            zip_filename = f"{os.path.splitext(file1)[0]}_{os.path.splitext(file2)[0]}.zip"
            zip_path = os.path.join(output_folder, zip_filename)

            # Create a ZipFile and add the two files
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.write(os.path.join(folder_path, file1), os.path.basename(file1))
                zipf.write(os.path.join(folder_path, file2), os.path.basename(file2))

            print(f"Zipped files: {file1}, {file2} into {zip_filename}")

# Example usage
folder_path = 'kitty_open3d/output/003/003'
output_folder = 'kitty_open3d/output/003/003'
zip_every_two_files(folder_path, output_folder)
