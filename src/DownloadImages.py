from google_images_search import GoogleImagesSearch

# Specify Google API keys for google image search
gis = GoogleImagesSearch('your_dev_api_key', 'your_project_cx')

# Define query strings in google search
search_strings = [
    "car",
    "truck",
    "buss",
    "motorcycle",
    "bicycle",
    "fire hydrant",
    "stop sign",
    "dog",
    "cat"
]

# Location to store downloaded images
base_download_dir = '/Downloads'

# download_images expects a query string that is passed to google's image search API.
# 1,000 JPG images are downloaded into subdirectories for each query string.
def download_images(query):
    # Google specific search parameters
    search_params = {
        'q': query,
        'num': 1000,
        'safe': 'high',
        'fileType': 'jpg',
        'imgType': 'photo'
    }

    try:
        gis.search(search_params=search_params, path_to_dir=base_download_dir + str(query))
    except:
        print("Cannot download images from query: " + str(query))

# main function
def main():
    for query in search_strings:
        download_images(query)

if __name__ == "__main__":
    main()