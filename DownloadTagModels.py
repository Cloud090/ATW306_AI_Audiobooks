import gdown

FOLDERS = [
    "https://drive.google.com/drive/folders/1SLzmeVd7MiGdleE08B8M277nhkJI5KjG",
    "https://drive.google.com/drive/folders/1sblJTtnUY5xVPSoF0g5YaZ1vK1-z_g63",
    "https://drive.google.com/drive/folders/13FX6G6D_gGsZJsr5vZkeZ9omLJ3IYze6"
]

def download_folder(url):
    print(f"\nDownloading: {url}")
    gdown.download_folder(
        url=url,
        quiet=False,
        use_cookies=False
    )
    print("Done.\n")

for url in FOLDERS:
    download_folder(url)

print("All downloads complete.")
