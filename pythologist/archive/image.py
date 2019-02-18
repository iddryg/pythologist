import xml.etree.ElementTree as ET
from tifffile import TiffFile
from tifffile import imsave as imsave_tiff
from collections import OrderedDict
import pandas as pd
import numpy as np
from io import BytesIO
from imageio import imsave
import sys, os, re, h5py
import matplotlib.pyplot as plt
from PIL import Image
from multiprocessing import Pool, cpu_count
from uuid import uuid4

class ComponentImageFrame(pd.DataFrame):
    _metadata = ['_images']
    @property
    def _constructor(self):
        return ComponentImageFrame
    def __init__(self,*args,**kw):
        # hide mpp for now when we send to parent since it doesn't use that argument
        super(ComponentImageFrame,self).__init__(*args,**kw)
    def _clean_extra_images(self):
        # Remove images that aren't in the dataframe
        image_ids = set(list(self['image_id'].unique()))
        spillover = list(set(list(self._images.keys()))-image_ids)
        for image_id in spillover:
            del self._images[image_id]
    def copy(self):
        v = ComponentImageFrame(pd.DataFrame(self).copy())
        v._images = self._images.copy()
        v._clean_extra_images() 
        return v
    def to_hdf(self,path,mode='r+'):
        pd.DataFrame(self).to_hdf(path,
                                  'component_image_frame',
                                  mode=mode,
                                  format='table',
                                  complib='zlib',
                                  complevel=9)
        f = h5py.File(path,'r+')
        for myid in self._images:
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            dset = f.create_dataset(myid, (1,), dtype=dt)
            dset[0] = np.fromstring(self._images[myid], dtype='uint8')
    def read(self,path,verbose=False,limit=None,threads=0):
        images = OrderedDict()
        base = os.path.abspath(path)
        path = os.path.abspath(path)
        frames = []
        z = 0
        for p, dirs, files in os.walk(path,followlinks=True,topdown=False):
            mydir = p[len(path):].strip('/')
            segs = [x for x in files if re.search('_component_data.tif$',x)]
            if len(segs) == 0: continue
            #s = Sample(p,mydir,verbose)
            for seg in segs:
                z += 1
                #frames.append((mydir,os.path.basename(p),seg))
                frame = re.match('(.+)_component_data.tif$',seg).group(1)
                frames.append((mydir,os.path.basename(p),frame,os.path.join(p,seg)))
                if limit is not None and z > limit: break
            if limit is not None and z > limit: break
        if threads == 0: threads=cpu_count()
        pool = Pool(processes=threads)
        results = pool.imap_unordered(ComponentImageFrame._read_frame,frames)
        output = []
        for result in results:
            folder = result['folder']
            sample = result['sample']
            frame = result['frame']
            for name in result['ci']:
                image = result['ci'][name]['image']
                id = str(uuid4())
                images[id] = image
                s = pd.Series(OrderedDict({
                    'folder':folder,
                    'sample':sample,
                    'frame':frame,
                    'stain':name,
                    'image_id':id
                }))
                output.append(s)
        pool.close()
        df = pd.DataFrame(output)
        super(ComponentImageFrame,self).__init__(df)
        self._images = images
    @staticmethod
    def _read_frame(vars):
        folder = vars[0]
        sample = vars[1]
        frame = vars[2]
        path = vars[3]
        ci = read_component_image(path).to_dict()
        return OrderedDict({
            'folder':folder,
            'sample':sample,
            'frame':frame,
            'ci':ci
        })

def read_component_image(input_tiff,quality=50,verbose=False):
    ci = ComponentImage()
    ci.read_image(input_tiff,quality=quality,verbose=verbose)
    return ci

class GenericImage:
    def __init__(self,image_dict=None):
        if image_dict is None:
            self._images = OrderedDict()
        else: self._images = image_dict
    def to_dict(self):
        return self._images
    def _show_image(self,img_data):
        plt.figure(figsize=(50,50))
        plt.axis("off")
        img = Image.open(img_data)
        plt.imshow(np.asarray(img), interpolation='nearest')
        plt.show()

class ComponentImage(GenericImage):
    def __init__(self,image_dict=None):
        super().__init__(image_dict)
    def __repr__(self):
        for name in self._images:
            self._show_image(BytesIO(self._images[name]['image']))
        return ''
    def read_image(self,input_tiff,quality=50,verbose=False):
        frame_images = OrderedDict()
        with TiffFile(input_tiff) as tif:
            images = tif.asarray()
            for page in tif.pages:
                name = None
                description = None
                if 'image_description' in page.tags.keys():
                    description = page.tags['image_description'].value.decode('utf-8')
                    tree = ET.ElementTree(ET.fromstring(description)).getroot()
                    names = [x for x in tree if x.tag == 'Name']
                    if len(names) > 0:
                        name = names[0].text
                img_data = page.asarray()
                if len(img_data.shape) > 2: continue # skip anything thats not single channel
                if name is None: name = "unknown"
                if verbose: sys.stderr.write(name+"\n")
                frame_images[name] = OrderedDict()
                frame_images[name]['description'] = description
                frame_images[name]['image'] = img_data
                df = pd.DataFrame(frame_images[name]['image'])
                rng = (df.min().min(),df.max().max())
                d = df.applymap(lambda x: 255*(x-rng[0])/(rng[1]-rng[0])).\
                    applymap(np.uint8).as_matrix().astype(np.uint8)
                bs = BytesIO()
                imsave(bs,d,format='jpeg',quality=quality)
                frame_images[name]['image'] = bs.getvalue()
        self._images = frame_images

def read_binary_seg_map_image(input_tiff,verbose=False):
    ci = BinarySegMapImage()
    ci.read_image(input_tiff,verbose=verbose)
    return ci

class BinarySegMapImage(GenericImage):
    def __init__(self,image_dict=None):
        super().__init__(image_dict)
    def __repr__(self):
        if 'ProcessRegionImage' in self._images:
            print('ProcessRegionImage')
            self._show_image(BytesIO(self._images['ProcessRegionImage']['image']))
        if 'SegmentationImage' in self._images:
            for compartment in self._images['SegmentationImage']:
                print(('SegmentationImage',compartment))
                self._show_image(BytesIO(self._images['SegmentationImage'][compartment]['image']))
        return ''
    def read_image(self,input_tiff,verbose=False):
        frame_images = OrderedDict()
        with TiffFile(input_tiff) as tif:
            images = tif.asarray()
            for page in tif.pages:
                compartment = None
                description = None
                kind = None
                if 'image_description' in page.tags:
                    description = page.tags['image_description'].value #.decode('utf-8')
                elif 'ImageDescription' in page.tags:
                    description = page.tags['ImageDescription'].value #.decode('utf-8')
                tree = ET.ElementTree(ET.fromstring(description)).getroot()
                kind = tree.tag
                compartments = [x.text for x in tree if x.tag == 'CompartmentType']
                if len(compartments) > 0: compartment = compartments[0]
                data = page.asarray()
                df = pd.DataFrame(data)
                rng = (df.min().min(),df.max().max())
                d = df.applymap(lambda x: 255*(x-rng[0])/(rng[1]-rng[0])).\
                    applymap(np.uint8).as_matrix().astype(np.uint8)
                bs = BytesIO()
                imsave_tiff(bs,d,compress=9)
                if kind not in frame_images:
                    frame_images[kind] = OrderedDict()
                if kind == 'ProcessRegionImage':
                    frame_images[kind]['description'] = description
                    frame_images[kind]['image'] = bs.getvalue()
                if kind == 'SegmentationImage':
                    frame_images[kind][compartment] = OrderedDict()
                    frame_images[kind][compartment]['description'] = description
                    frame_images[kind][compartment]['image'] = bs.getvalue()
        self._images = frame_images
