from tifffile import TiffFile
import numpy as np
import pandas as pd
def read_tiff_stack(filename):
    data = []
    with TiffFile(filename) as tif:
        image_stack = tif.asarray()
        for page in tif.pages:
            meta = dict((tag.name,tag.value) for tag in page.tags.values())
            data.append({'raw_meta':meta,'raw_image':np.array(page.asarray())})
    return data

def flood_fill(image,x,y,exit_criteria,max_depth=1000,visited=None,border_trim=1):
    # return a list of coordinates we fill without visiting twice or hitting an exit condition
    if visited is None: visited = set()
    if len(visited)>=max_depth: return visited
    if y < 0+border_trim or y >= image.shape[0]-border_trim: return visited
    if x < 0+border_trim or x >= image.shape[1]-border_trim: return visited
    if (x,y) in visited: return visited
    if exit_criteria(image[y][x]): 
        return visited
    visited.add((x,y))
    # traverse deeper
    visited = flood_fill(image,x,y+1,exit_criteria,max_depth=max_depth,visited=visited)
    visited = flood_fill(image,x+1,y,exit_criteria,max_depth=max_depth,visited=visited)
    visited = flood_fill(image,x,y-1,exit_criteria,max_depth=max_depth,visited=visited)
    visited = flood_fill(image,x-1,y,exit_criteria,max_depth=max_depth,visited=visited)
    return visited

def map_image_ids(image):
    nmap = pd.DataFrame(image.astype(float)).stack().reset_index().\
       rename(columns={'level_0':'y','level_1':'x',0:'id'}).\
       query('id!=0')
    nmap.loc[~np.isfinite(nmap['id']),'id'] = 0
    nmap['id'] = nmap['id'].astype(int)
    return nmap[['x','y','id']]