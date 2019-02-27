import pandas as pd
import numpy as np
from pythologist.measurements import Measurement
from pythologist_reader.utilities import watershed_image, map_image_ids
import sys, os, io
import imageio
class Images(Measurement):
    def __init__(self,*args,**kwargs):
        super(Images,self).__init__(*args,**kwargs)
        self._edge_map_cache = None
        self._cell_map_cache = None
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
        return base.merge(pd.DataFrame(data,columns=['sample_id','frame_id','shape']),on=['sample_id','frame_id'])

    def get_outline_images(self,subset_logic=None,edge_color=(0,0,255,255),fill_color=(135,206,250,255)):
        v = self.get_segmentation_map_images(type='edge',subset_logic=subset_logic,color=edge_color,blank=(0,0,0,0),watershed_steps=1).\
            rename(columns={'image':'edge'}).\
            merge(self.get_segmentation_map_images(type='cell',subset_logic=subset_logic,color=fill_color,blank=(0,0,0,0)),on=list(self.columns)).\
            rename(columns={'image':'cell'})
        v['merged'] = v.apply(lambda x: _merge_images(x['cell'],x['edge']),1)
        return v

    def get_segmentation_map_images(self,type='edge',subset_logic=None,color=None,watershed_steps=0,blank=(0,0,0,255)):
        # if subset logic is set only plot those cells
        # if color is set color all cells that color
        #if os.path.exists(path) and overwrite is False: raise ValueError("Error: use ovewrite=True to overwrite images")
        #if not os.path.exists(path): os.makedirs(path)
        ems = self.get_segmentation_maps(type=type)
        if subset_logic is not None: 
            subset = self.cdf.subset(subset_logic)
            ems = ems.merge(subset.loc[:,subset.frame_columns+['cell_index']],on=subset.frame_columns+['cell_index'])
        edf = ems.set_index(list(self.columns))
        imgs = []
        for i,r in self.iterrows():
            edfsub = edf.loc[tuple(r)]
            imsize = r['shape']
            img = pd.DataFrame(np.zeros(imsize))
            for i2,r2 in edfsub.iterrows():
                img[r2['x']][r2['y']] = r2['cell_index']
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
                #print(type(img))
                #imgdf = pd.DataFrame(img).applymap(lambda x: blank if x ==0 else color)

            imgs.append(list(r)+[img])
        return pd.DataFrame(imgs,columns=list(self.columns)+['image'])

    def get_segmentation_maps(self,type='edge'):
        if type == 'edge' and self._edge_map_cache is not None: return self._edge_map_cache
        if type == 'cell' and self._cell_map_cache is not None: return self._cell_map_cache
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
