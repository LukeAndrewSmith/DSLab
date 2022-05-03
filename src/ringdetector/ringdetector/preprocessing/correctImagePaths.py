# script to align the image paths in the labelme jsons:
import os
import json
from ringdetector.Paths import DATA, LABELME_JSONS

for file in os.listdir(os.path.join(LABELME_JSONS)):
    if file.endswith(".json"):
        f = open(os.path.join(LABELME_JSONS,file))
        data = json.load(f)
        im_path = data["imagePath"]
        im_name = im_path.split('/')[-1]
        new_path = f'../images/{im_name}'

        data["imagePath"] = new_path
        print(data)


        if im_path != new_path:
            print("======")
            print(im_path)
            print(new_path)
            with open(os.path.join(LABELME_JSONS,file), 'w') as nf:
                json.dump(data, nf)



