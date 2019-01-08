from tifffile import TiffFile
import numpy as np
def read_tiff_stack(filename):
    data = []
    with TiffFile(filename) as tif:
        image_stack = tif.asarray()
        for page in tif.pages:
            meta = dict((tag.name,tag.value) for tag in page.tags.values())
            data.append({'raw_meta':meta,'raw_image':np.array(page.asarray())})
    return data