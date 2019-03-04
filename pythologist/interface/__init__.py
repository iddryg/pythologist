import pandas as pd
import numpy as np
from pythologist.measurements import Measurement
from pythologist_image_utilities import watershed_image, map_image_ids
import sys, os, io
import imageio
from PIL import Image

class SegmentationImageOutput(pd.DataFrame):
    _metadata = []
    def __init__(self,*args, **kw):
        super(SegmentationImageOutput, self).__init__(*args, **kw) 
        if 'verbose' in kw: self.verbose = kw['verbose']
    @property
    def _constructor(self):
        return SegmentationImageOutput
    def write_to_path(self,path,suffix='',format='png',overwrite=False):
        """
        Output the data the dataframe's 'image' column to a directory structured by project->sample and named by frame
        """
        if os.path.exists(path) and overwrite is False: raise ValueError("Error: use ovewrite=True to overwrite images")
        if not os.path.exists(path): os.makedirs(path)
        for i,r in self.iterrows():
            spath = os.path.join(path,r['project_name'],r['sample_name'])
            if not os.path.exists(spath): os.makedirs(spath)
            if suffix == '':
                fname = os.path.join(spath,r['frame_name']+'.'+format)
            else: fname = os.path.join(spath,r['frame_name']+'_'+suffix+'.'+format)
            imageio.imwrite(fname, r['image'],format=format)

class SegmentationImages(Measurement):
    def __init__(self,*args,**kwargs):
        super(SegmentationImages,self).__init__(*args,**kwargs)
        self._edge_map_cache = None
        self._cell_map_cache = None
        self._coordinates = None
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        base = pd.DataFrame(cdf).copy().loc[:,cdf.frame_columns].drop_duplicates()
        data = []
        if 'verbose' in kwargs and kwargs['verbose']: sys.stderr.write("Reading image names/ids and sizes\n")
        for sample_id in base['sample_id'].unique().tolist():
            if 'verbose' in kwargs and kwargs['verbose']: sys.stderr.write("Reading sample "+str(sample_id)+"\n")
            s = cdf.db.get_sample(sample_id)
            for frame_id in base.loc[base['sample_id']==sample_id,'frame_id']:
                f = s.get_frame(frame_id)
                data.append([sample_id,frame_id,f.shape])


        base = base.merge(pd.DataFrame(data,columns=['sample_id','frame_id','shape']),on=['sample_id','frame_id'])
        return base

    def write_segmentation_images(self,schema,path,suffix='',format='png',overwrite=False,background=(0,0,0,255)):
        """
        executes the build_segmentation_image and write_segmentation_image for each frame
                 rather than executing on all frames at once.
        """
        if os.path.exists(path) and overwrite is False: raise ValueError("Error: use ovewrite=True to overwrite images")
        for i,r in self.iterrows():
            subcdf = self.cdf.loc[self.cdf['frame_id']==r['frame_id']].copy()
            if self.verbose: sys.stderr.write("==============\nWriting schema for: \n"+str(r)+"\n\n")
            subsegs = subcdf.segmentation_images(verbose=self.verbose)
            layers = subsegs.build_segmentation_image(schema,background=background)
            layers.write_to_path(path,overwrite=True)
            if self.verbose: sys.stderr.write("Finished writing schema.\n")



    def get_coordinates(self):
        if self._coordinates is not None: return self._coordinates
        df = self.set_index(self.cdf.frame_columns+['shape']).stack().\
            reset_index().rename(columns={'level_7':'image_type',0:'image'}).\
            set_index(self.cdf.frame_columns+['shape','image_type'])
        imgs = []
        for i,r in df.iterrows():
            if self.verbose: sys.stderr.write("Extracting coordinates from "+str(list(i))+"\n")
            left = pd.DataFrame([i],columns=df.index.names)
            left['_key'] = 1
            img=map_image_ids(r['image']).groupby('id').apply(lambda x: list(zip(*x[['x','y']].apply(tuple).tolist())))
            img = img.reset_index().rename(columns={'id':'cell_index',0:'coords'})
            img['_key'] = 1
            img = left.merge(img,on='_key').drop(columns='_key')
            imgs.append(img)
        imgs = pd.concat(imgs)
        self._coordinates = imgs
        return imgs

    def get_segmentation_map_images(self,type='edge',subset_logic=None,color=None,watershed_steps=0,blank=(0,0,0,255)):
        # if subset logic is set only plot those cells
        # if color is set color all cells that color
        if self.verbose: sys.stderr.write("getting segmap "+str(type)+"\n")
        ems = self.get_segmentation_maps(type=type)
        if subset_logic is not None: 

            subset = self.cdf.subset(subset_logic)
            ems = ems.merge(subset.loc[:,subset.frame_columns+['cell_index']],on=subset.frame_columns+['cell_index'])
        edf = ems.set_index(list(self.columns))
        imgs = []
        for i,r in self.iterrows():
            imsize = r['shape']
            img = pd.DataFrame(np.zeros(imsize))
            if subset.shape[0] == 0: # case where there is nothing to do
                if self.verbose: sys.stderr.write("Empty image for this phenotype subset\n")
                imgs.append(list(r)+[img])
                continue
            edfsub = edf.loc[tuple(r)]
            #if self.verbose: sys.stderr.write("make image and fill zeros\n")
            fullx = pd.DataFrame({'x':list(range(0,imsize[1]))})
            fullx['_key']=1
            fully = pd.DataFrame({'y':list(range(0,imsize[0]))})
            fully['_key']=1
            full = fullx.merge(fully,on='_key').merge(edfsub,on=['x','y'],how='left').fillna(0)
            img = np.array(full.pivot(columns='x',index='y',values='cell_index').astype(int))
            #if self.verbose: sys.stderr.write("finished making image and fill zeros\n")
            if watershed_steps > 0:
                # get the zero and nonzero components
                mid = map_image_ids(img,remove_zero=False)
                midzero = list(zip(*mid.query('id==0').copy()[['x','y']].apply(tuple).tolist()))
                mid = list(zip(*mid.query('id!=0').copy()[['x','y']].apply(tuple).tolist()))
                img = watershed_image(np.array(img),mid,midzero,steps=watershed_steps)
            if color is not None:
                # we need to make a new image thats colored in
                fresh = np.zeros(list(imsize)+[len(color)]).astype(int)
                blank = tuple(list(blank)[0:len(color)])
                fresh[:][:] = blank
                # get our coordinates
                coords = np.array(list(zip(*map_image_ids(img)[['y','x']].apply(tuple).tolist())))
                fresh[tuple([*coords.T])] = color
                #for i2,r2 in map_image_ids(img).iterrows():
                #    fresh[r2['y']][r2['x']] = color
                img = fresh
            imgs.append(list(r)+[img])
        imgs = pd.DataFrame(imgs,columns=list(self.columns)+['image'])
        return imgs


    def build_segmentation_image(self,schema,background=(0,0,0,0)):
        """
        Put together an image.  Defined by a list of layers with RGBA colors

        Example:
            schema = [
                {'subset_logic':SL(phenotypes=['SOX10+']),
                 'edge_color':(31, 31, 46,255),
                 'watershed_steps':0,
                 'fill_color':(51, 51, 77,255)
                },
                {'subset_logic':SL(phenotypes=['CD8+'],scored_calls={'PD1':'+'}),
                 'edge_color':(255,0,0,255),
                 'watershed_steps':1,
                 'fill_color':(0,0,0,255)
                },
                {'subset_logic':SL(phenotypes=['CD8+'],scored_calls={'PD1':'-'}),
                 'edge_color':(255,0,255,255),
                 'watershed_steps':1,
                 'fill_color':(0,0,255,255)
                }
            ]
            imgs = imageaccess.build_segmentation_image(schema,background=(0,0,0,255))
        """
        cummulative = self.copy()
        def _set_blank(img,blank):
            img[:][:] = blank
            return img
        cummulative['merged'] = cummulative.apply(lambda x: 
            _set_blank(np.zeros(list(x['shape'])+[4]),background)
            ,1)
        for layer in schema:
            if self.verbose: sys.stderr.write("Calculating layer "+str(layer)+"\n")
            images  = self.get_outline_images(subset_logic=layer['subset_logic'],
                                              edge_color=layer['edge_color'],
                                              watershed_steps=layer['watershed_steps'],
                                              fill_color=layer['fill_color'])
            cummulative = cummulative.rename(columns={'merged':'old'})
            cummulative = cummulative.merge(images,on=list(self.columns))
            cummulative['new'] = cummulative.apply(lambda x: _merge_images(x['merged'],x['old']),1)
            cummulative = cummulative.drop(columns=['old','merged']).rename(columns={'new':'merged'})
        cummulative = cummulative.rename(columns={'merged':'image'})
        return SegmentationImageOutput(cummulative)

    def get_outline_images(self,subset_logic=None,edge_color=(0,0,255,255),fill_color=(135,206,250,255),watershed_steps=1):
        if len(edge_color) == 3: edge_color = tuple(list(edge_color)+[255])
        if len(fill_color) == 3: fill_color = tuple(list(fill_color)+[255])
        #if self.verbose: sys.stderr.write("getting outline image\n")
        #if self.verbose: sys.stderr.write("reading edges\n")
        edge_images = self.get_segmentation_map_images(type='edge',subset_logic=subset_logic,color=edge_color,blank=(0,0,0,0),watershed_steps=watershed_steps).\
            rename(columns={'image':'edge'})
        if self.verbose: sys.stderr.write("reading cells\n")
        cell_images = self.get_segmentation_map_images(type='cell',subset_logic=subset_logic,color=fill_color,blank=(0,0,0,0)).\
            rename(columns={'image':'cell'})
        if self.verbose: sys.stderr.write("merge edge and cell\n")
        v = edge_images.merge(cell_images,on=list(self.columns))
        v['merged'] = v.apply(lambda x: _merge_images(x['edge'],x['cell']),1)
        #if self.verbose: sys.stderr.write("finished outline image\n")
        return v.drop(columns=['cell','edge'])


    def get_segmentation_maps(self,type='edge'):
        if type == 'edge' and self._edge_map_cache is not None: return self._edge_map_cache
        if type == 'cell' and self._cell_map_cache is not None: return self._cell_map_cache
        if self.verbose: sys.stderr.write("The "+str(type)+" map has not been calculated yet. ... computing.\n")
        outputs = self.apply_frames(lambda x: x.edge_map() if type == 'edge' else x.cell_map())
        dfs = []
        for i,r in outputs.iterrows():
            df = r['output']
            df.index = pd.Index([i for temp in range(0,r['output'].shape[0])])
            df.index.set_names(list(self.columns),inplace=True)
            dfs.append(df)
        dfs = pd.concat(dfs).reset_index()
        # we don't need every cell.. just the ones in this cell data frame
        dfs = dfs.merge(self.cdf[list(self.cdf.frame_columns)+['cell_index']],on=list(self.cdf.frame_columns)+['cell_index'])
        if type == 'edge': self._edge_map_cache = dfs
        elif type == 'cell': self._cell_map_cache = dfs
        else: raise ValueError('edge or cell')
        if self.verbose: sys.stderr.write("The "+str(type)+" map is finished.\n")
        return dfs

    def apply_frames(self,func):
        samples = self['sample_id'].unique().tolist()
        data = []
        for sample_id in samples:
            s = self.cdf.db.get_sample(sample_id)
            if self.verbose: sys.stderr.write("Read in sample "+s.sample_name+" ("+str(sample_id)+")\n")
            for frame_id in self.loc[self['sample_id']==sample_id,'frame_id']:
                f = s.get_frame(frame_id)
                data.append([sample_id,frame_id,func(f)])
        return self.merge(pd.DataFrame(data,columns=['sample_id','frame_id','output']),on=['sample_id','frame_id']).set_index(list(self.columns))

def _merge_images(image1,image2):
    edge = np.uint8(image1)
    cell = np.uint8(image2)
    bedge = io.BytesIO()
    bcell = io.BytesIO()
    imageio.imwrite(bedge, edge,format='tif')
    imageio.imwrite(bcell, cell,format='tif')
    pedge = Image.open(bedge)
    pcell = Image.open(bcell)
    pcell.paste(pedge, (0, 0), pedge)
    return np.array(pcell)

