'''

Tải dataset flickr30k từ Kaggle và giải nén nó vào thư mục data/flickr30k.

chạy lệnh: kaggle datasets download -d adityajn105/flickr30k

'''

import zipfile

with zipfile.ZipFile("flickr30k.zip", "r") as z:
    z.extractall("data/flickr30k")

print("Done")