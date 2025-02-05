'''
This script provides functionality to download data files required by Fmask. 
It includes functions to download individual files and a command-line interface to download all required data files.
Functions:
    download_data_from_google_drive(source_url, destination):
    download_data_from_drive(source_url, destination, drive='google_drive'):
    download_all_data(update):
Usage:
    Run the script with the command-line interface to download all required data files:
    $ python download_package_data.py --update [y/n]
'''

import os
import requests
import click
import constant as C

def download_data_from_google_drive(source_url, destination):
    """
    Downloads a file from Google Drive given its URL and saves it to the specified destination.
    Args:
        source_url (str): The URL of the file on Google Drive.
        destination (str): The local file path where the downloaded file will be saved.
    Raises:
        requests.exceptions.RequestException: If there is an issue with the network request.
    Example:
        download_data_from_google_drive(
            "https://drive.google.com/file/d/1A2B3C4D5E6F7G8H9I/view?usp=sharing",
            "/path/to/local/file"
        )
    """
    
    # Extract file ID from the Google Drive URL
    file_id = source_url.split("/d/")[1].split("/")[0]
    
    # Construct the download URL
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

    # Send a GET request to the URL to start the file download
    session = requests.Session()
    response = session.get(download_url, params={'id': file_id}, stream=True)

    # Write the content to the output file
    with open(destination, 'wb') as f:
        # For most use cases, 8192 bytes is a reasonable default. If you're dealing with very large files and have sufficient memory, you could experiment with a larger chunk size for potentially faster downloads.
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def download_data_from_drive(source_url, destination, drive='google_drive'):
    """
    Downloads data from a specified drive.
    Parameters:
    source_url (str): The URL of the data to be downloaded.
    destination (str): The local path where the downloaded data will be saved.
    drive (str, optional): The type of drive to download from. Default is 'google_drive'.
    Returns:
    None
    """
    
    if drive == 'google_drive':
        download_data_from_google_drive(source_url, destination)

@click.command()
@click.option('--update', default='n', help='Force update of the data files. y/n')
def download_all_data(update):
    """
    Downloads all data files (global and model) from the predefined URLs and saves them to the specified directory.
    """
    # obtain the current script directory's path with parent directory
    pacakge_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(pacakge_dir, 'data1')
    model_dir = os.path.join(pacakge_dir, 'model1')
    
    # create the folder if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Download global data
    if (not os.path.exists(os.path.join(data_dir, 'global_gswo150.tif')) or update == 'y'):
        download_data_from_drive(C.URL_global_gswo150, os.path.join(data_dir, 'global_gswo150.tif'))
        print("Downloaded global water layer")
    else:
        print("Global water layer already exists")
    
    if (not os.path.exists(os.path.join(data_dir, 'global_gt30.tif'))or update == 'y'):
        download_data_from_drive(C.URL_global_gt30, os.path.join(data_dir, 'global_gt30.tif'))
        print("Downloaded global DEM")
    else:
        print("Global DEM already exists")
        
    if (not os.path.exists(os.path.join(model_dir, 'unet_ncf_l7.pt'))or update == 'y'):
        download_data_from_drive(C.URL_unet_ncf_l7, os.path.join(model_dir, 'unet_ncf_l7.pt'))
        print("Downloaded base UNet model for Landsat 4-7")
    else:
        print("Base UNet model for Landsat 4-7 already exists")
    
    if (not os.path.exists(os.path.join(model_dir, 'unet_ncf_l8.pt'))or update == 'y'):
        download_data_from_drive(C.URL_unet_ncf_l8, os.path.join(model_dir, 'unet_ncf_l8.pt'))
        print("Downloaded base UNet model for Landsat 8-9")
    else:
        print("Base UNet model for Landsat 8-9 already exists")
    
    if (not os.path.exists(os.path.join(model_dir, 'unet_ncf_s2.pt')) or update == 'y'):
        download_data_from_drive(C.URL_unet_ncf_s2, os.path.join(model_dir, 'unet_ncf_s2.pt'))
        print("Downloaded base UNet model for Sentinel-2")
    else:
        print("Base UNet model for Sentinel-2 already exists")

if __name__ == '__main__':
    download_all_data()