import pandas as pd
import numpy as np
from pythologist.measurements import Measurement
from pythologist_image_utilities import watershed_image, map_image_ids, generate_new_region_image
import sys, os, io
import imageio
from PIL import Image
from pythologist import SubsetLogic as SL
from scipy.ndimage import gaussian_filter
from tempfile import TemporaryDirectory, NamedTemporaryFile
from skimage import img_as_ubyte

class SegmentationImageOutput(pd.DataFrame):
    """
    The Segmentation Image Output class 
    """
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

        Args:
            path (str): Where to write the directory of images
            suffix (str): for labeling the imaages you write
            format (str): default 'png' format to write the file
            overwrite (bool): default False. if true can overwrite files in the path

        Modifies:
            Creates path folder if necessary and writes images to path
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
    """
    Class suitable for generating image outputs
    """
    def __init__(self,*args,**kwargs):
        """
        Args:
            verbose (bool): output more details if True
        """
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
        if self.verbose: sys.stderr.write("getting segmap "+str(type)+"\n")
        ems = self.get_segmentation_maps(type=type)
        subset = self.cdf
        if subset_logic is not None: 
            subset = self.cdf.subset(subset_logic)
            ems = ems.merge(subset.loc[:,subset.frame_columns+['cell_index']],on=subset.frame_columns+['cell_index'])
        #edf = ems.set_index(list(self.columns))
        imgs = []
        for i,r in self.iterrows():
            imsize = r['shape']
            img = pd.DataFrame(np.zeros(imsize))
            if subset.shape[0] == 0: # case where there is nothing to do
                if self.verbose: sys.stderr.write("Empty image for this phenotype subset\n")
                imgs.append(list(r)+[np.array(img)])
                continue
            #edfsub = edf.loc[tuple(r)]
            edfsub = ems.loc[ems['frame_id']==r['frame_id']].copy().set_index(list(self.columns))

            #if self.verbose: sys.stderr.write("make image and fill zeros\n")
            fullx = pd.DataFrame({'x':list(range(0,imsize[1]))})
            fullx['_key']=1
            fully = pd.DataFrame({'y':list(range(0,imsize[0]))})
            fully['_key']=1
            full = fullx.merge(fully,on='_key').merge(edfsub,on=['x','y'],how='left').fillna(0)
            img = np.array(full.pivot(columns='x',index='y',values='cell_index').astype(int))
            if map_image_ids(img).shape[0] == 0:
                # There is nothing for us to draw with this phenotype and image
                imgs.append(list(r)+[np.zeros(imsize)])
                continue
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

        Make the schema example
        
        |    schema = [
        |        {'subset_logic':SL(phenotypes=['SOX10+']),
        |         'edge_color':(31, 31, 46,255),
        |         'watershed_steps':0,
        |         'fill_color':(51, 51, 77,255)
        |        },
        |        {'subset_logic':SL(phenotypes=['CD8+'],scored_calls={'PD1':'+'}),
        |         'edge_color':(255,0,0,255),
        |         'watershed_steps':1,
        |         'fill_color':(0,0,0,255)
        |        },
        |        {'subset_logic':SL(phenotypes=['CD8+'],scored_calls={'PD1':'-'}),
        |         'edge_color':(255,0,255,255),
        |         'watershed_steps':1,
        |         'fill_color':(0,0,255,255)
        |        }
        |    ]
        |    imgs = imageaccess.build_segmentation_image(schema,background=(0,0,0,255))
        

        Args:
            schema (list): a list of layers (see example above)
            background (tuple): a color RGBA 0-255 tuple for the. background color
        Returns:
            SegmentationImageOutput: an output suitable for writing images
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


def _get_new_regions(cdf,sample,frame_id,unset_label='undefined',gaussian_sigma=66,gaussian_truncate=4,verbose=False):
    sub = cdf.loc[cdf['frame_id']==frame_id]
    #print(sub.iloc[0][['project_name','sample_name','frame_name']])
    #print(sub.shape)
    shape = sub.iloc[0]['frame_shape']
    dfs = {}
    sid = sub.iloc[0]['sample_id']
    fid = sub.iloc[0]['frame_id']
    proc = cdf.db.get_sample(sid).get_frame(fid).processed_image
    present = sub['phenotype_label'].unique()
    for p in sub.phenotypes:
        empty = np.zeros(shape)
        if p not in present:
            dfs[p] = empty.copy().astype(float)
            continue
        emap = pd.DataFrame(empty).stack().reset_index().astype(int)
        emap.columns = ['y','x','id']
        emap = emap.drop(columns='id')
        
        sel = sub.subset(SL(phenotypes=[p]))[['x','y']].drop_duplicates()
        sel['id'] = 1
        sel = emap.merge(sel,on=['x','y'],how='left').fillna(0).pivot(columns='x',index='y',values='id')
        sel = np.array(sel).astype(float)
        blur = gaussian_filter(sel,gaussian_sigma,truncate=gaussian_truncate)
        dfs[p] = blur
    regions = {}
    
    #print('mask')
    remainder = np.ones(shape).astype(bool)
    for p in dfs:
        others = set(dfs.keys())-set([p])
        result = np.ones(dfs[p].shape).astype(bool)
        for o in others:
            result = (dfs[p] > dfs[o])&result&proc
        #print(p)
        result = result.astype(np.uint8)
        remainder = remainder&(~result.astype(bool))
        regions[p] = result.astype(np.uint8)
    regions[unset_label] = (remainder&proc).astype(np.uint8)
    return regions

def _extract_regionization_features(cpi,cdf,sample_id,frame_id):
    frame = cpi.get_sample(sample_id).get_frame(frame_id)
    processed_image = frame.get_image(frame.processed_image_id)
    segmentation_image = frame.get_image(frame.get_data('segmentation_images').\
                                             set_index('segmentation_label').\
                                             loc['cell_map','image_id'])
    label_df = cdf.loc[cdf['frame_id']==frame_id].loc[:,['cell_index','phenotype_label']].\
        set_index('cell_index')
    label_dict = label_df.loc[~label_df['phenotype_label'].isna(),'phenotype_label'].to_dict()
    return processed_image, segmentation_image,label_dict
def phenotypes_to_regions(cdf,path=None,
                          gaussian_sigma=66,
                          gaussian_truncate=4,
                          gaussian_mode='reflect',
                          verbose=False,
                          overwrite=False,
                          tempdir=None
                        ):
    if cdf.db is None: raise ValueError("cannot execute without the source CellProject db attribute set")
    if path and os.path.exists(path) and not overwrite: raise ValueError("cannot overwrite unless overwrite is True")
    if cdf['project_name'].unique().shape[0] > 1: raise ValueError("cannot do this with multiple project names")
    if cdf['project_id'].unique().shape[0] > 1: raise ValueError("cannot do this with multiple project ids")
    if path is None:
        tnf = NamedTemporaryFile(delete=False,dir=tempdir)
    output = cdf.db.__class__(path if path is not None else tnf.name,mode='w')
    output.project_name = cdf.iloc[0]['project_name']
    for sample_id in cdf['sample_id'].unique():
        sample = cdf.loc[cdf['sample_id']==sample_id,:]
        if verbose: sys.stderr.write("Doing sample "+str((sample_id,sample.iloc[0]['sample_name']))+"\n")
        sample_cpi = cdf.db.get_sample(sample_id)
        for frame_id in sample.loc[sample['sample_id']==sample_id,'frame_id'].unique():
            frame_cpi = sample_cpi.get_frame(frame_id)
            processed_image, segmentation_image,label_dict = _extract_regionization_features(cdf.db,cdf,sample_id,frame_id)
            new_region, region_key = generate_new_region_image(processed_image,
                                                               segmentation_image,
                                                               label_dict,
                                                               sigma=gaussian_sigma,
                                                               truncate=gaussian_truncate,
                                                               mode=gaussian_mode
                                                              )
            regions = {}
            for i, region in region_key.items():
                layer = np.zeros(processed_image.shape)
                layer[np.where(new_region==i)] = 1
                regions[region] = layer.astype(np.int8)
            frame_cpi.set_regions(regions,
                          use_processed_region=True,
                          unset_label='undefined',
                          verbose=verbose)
        output.append_sample(sample_cpi)
    output.set_id(cdf.iloc[0]['project_id'])
    if verbose: 
        sys.stderr.write("generate the output dataframe\n")
    _temp = output.cdf
    _temp = _temp[['sample_id','frame_id','region_label','regions','cell_index']].rename(columns={'region_label':'temp1','regions':'temp2'})
    cdf2 = cdf.merge(_temp,on=['sample_id','frame_id','cell_index'])
    cdf2['region_label'] = cdf2['temp1']
    cdf2['regions'] = cdf2['temp2']
    cdf2 = cdf2.drop(columns=['temp1','temp2'])
    #if path is None:
    #    os.remove(tnf.name)
    cdf2.db = output
    cdf2.microns_per_pixel = cdf.microns_per_pixel
    return cdf2, output
def get_region_images(cdf,output_path,colors,background_color='#000000',overwrite=False,format='png',verbose=False):
    def hex_to_rgb(h):
        h = h.lstrip('#')
        v =  tuple(list(int(h[i:i+2], 16) for i in (0, 2, 4))+[255])
        return [x/255 for x in v]

    def write_regions(frame,basedir,colors,background_color,format):
        rshape = frame.get_data('regions').iloc[0]['image_id']
        rshape = frame.get_image(rshape).shape
        start = np.zeros(list(rshape)+[4])
        start[:,:]=hex_to_rgb(background_color)
        fname = frame.frame_name
        for i,r in frame.get_data('regions').iterrows():
            col = colors[r['region_label']]
            img = frame.get_image(r['image_id'])
            start[img==1]= hex_to_rgb(col)
        imageio.imwrite(os.path.join(basedir,fname+'.'+format), start,format=format)

    if not cdf.db: raise ValueError("Need db set")
    if os.path.exists(output_path) and not overwrite: raise ValueError("overwrite is set to False")
    #os.makedirs(output_path)
    #for s in cdf.db.sample_iter():
    #    os.makedirs(os.path.join(output_path,s.sample_name))
    values = cdf.loc[:,['project_name','sample_id','frame_name','frame_id']].drop_duplicates()
    for pname in values['project_name'].unique():
        samples = values.loc[values['project_name']==pname]
        for sid in samples['sample_id'].unique():
            s = cdf.db.get_sample(sid)
            sname = s.sample_name
            if verbose: sys.stderr.write(str((pname,sname))+"\n")
            basedir = os.path.join(output_path,pname,sname)
            if not os.path.exists(basedir):
                os.makedirs(basedir)
            frames = samples.loc[samples['sample_id']==sid]
            for fid in frames['frame_id'].unique():
                write_regions(s.get_frame(fid),basedir,colors,background_color,format)

def fetch_single_segmentation_image_bytes(self,schema,background=(0,0,0,255),tempdir=None):
    if self['frame_name'].unique().shape[0]!=1: raise ValueError("must be only one frame name")
    if self['frame_id'].unique().shape[0]!=1: raise ValueError("must be only one frame id")
    sio = self.segmentation_images().build_segmentation_image(schema,background=(0,0,0,255))
    with TemporaryDirectory(dir=tempdir) as td:
        sio.write_to_path(td,overwrite=True,format='png')
        for base, dirs, files in os.walk(td):
            #print((base,dirs,files))
            for fname in files:
                #print(fname[-3:])
                return open(os.path.join(base,fname),'rb').read()

def fetch_single_region_image_bytes(self,colors,background='#000000',tempdir=None,verbose=False):
    if self['frame_name'].unique().shape[0]!=1: raise ValueError("must be only one frame name")
    if self['frame_id'].unique().shape[0]!=1: raise ValueError("must be only one frame id")
    with TemporaryDirectory(dir=tempdir) as td:
        outfile = get_region_images(self,td,colors,background_color=background,overwrite=True,format='png',verbose=verbose)
        for base, dirs, files in os.walk(td):
            for fname in files:
                return open(os.path.join(base,fname),'rb').read()