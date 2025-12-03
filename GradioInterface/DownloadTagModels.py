# This script downloads the current tagging models from Google Drive - emotion tagging, background sound effect tagging, and foreground sound effect tagging. It can be run using the "DownloadTagModels.bat" file.

import gdown

# Google drive folder locations.
FOLDERS = [
    "https://drive.google.com/drive/folders/1SLzmeVd7MiGdleE08B8M277nhkJI5KjG",
    "https://drive.google.com/drive/folders/1sblJTtnUY5xVPSoF0g5YaZ1vK1-z_g63",
    "https://drive.google.com/drive/folders/13FX6G6D_gGsZJsr5vZkeZ9omLJ3IYze6"
]

# Function to download a folder.
def download_folder(url):
    print(f"\nDownloading: {url}")
    gdown.download_folder(
        url=url,
        quiet=False,
        use_cookies=False
    )
    print("Done.\n")

# Download each folder in the list, using the above function.
for url in FOLDERS:
    download_folder(url)

print("All downloads complete.")
