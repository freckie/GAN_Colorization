from google_images_download import google_images_download

gid = google_images_download.googleimagesdownload()
args = {
    'keywords': 'Seoul',
    'limit': 100,
    'print_urls': False,
    'format': 'jpg',
    'size': '>400*300'
}
paths = gid.download(args)
print(paths)