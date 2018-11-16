import os, re, json, sys
from collections import OrderedDict
import pandas as pd
import numpy as np
from pythologist.formats import CellImageDataGeneric, CellImageSetGeneric
from uuid import uuid4

_float_decimals = 6

class CellImageSetInForm(CellImageSetGeneric):
    def __init__(self):
        super().__init__()
    def read_from_path(self,path,verbose=False,limit=None,sample_index=1):
        # Put together all the available sets of images recursing through the path
        #base = os.path.abspath(path)
        path = os.path.abspath(path)
        self._path = path
        rows = []
        z = 0
        for p, dirs, files in os.walk(self._path,followlinks=True,topdown=False):
            mydir = p[len(path):]
            z += 1
            segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
            if len(segs) == 0: continue
            if verbose: sys.stderr.write("SAMPLE: "+str(p)+"\n")
            #s = Sample(p,mydir,verbose,sample_index)
            #self._samples.append(s)
            #if limit is not None and z >= limit: break
            files = os.listdir(p)
            segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
            sample_folder = p.split(os.sep)[-1*sample_index] #os.path.basename(path)
            print(sample_folder)
            self._frames = OrderedDict()
            snames = set()
            for file in segs:
                m = re.match('(.*)cell_seg_data.txt$',file)
                score = os.path.join(p,m.group(1)+'score_data.txt')
                summary = os.path.join(p,m.group(1)+'cell_seg_data_summary.txt')
                binary_seg_maps = os.path.join(p,m.group(1)+'binary_seg_maps.tif')
                tfile = os.path.join(p,m.group(1)+'tissue_seg_data_summary.txt')
                tissue_seg_data = tfile if os.path.exists(tfile) else None
                sample = sample_folder
                snames.add(sample)
                frame = m.group(1).rstrip('_')
                data = os.path.join(p,file)
                if not os.path.exists(summary):
                    if verbose: sys.stderr.write('Missing summary file '+summary+"\n")
                    summary = None
                if not os.path.exists(score):
                    raise ValueError('Missing score file '+score)
                #self._frames[frame] = Frame(path,mydir,sample,frame,data,score,summary,binary_seg_maps,tissue_seg_data,verbose)
                if verbose: sys.stderr.write('Acquiring frame '+data+"\n")
                cid = CellImageDataInForm()
                cid.read_image_data(cell_seg_data_file=data,
                                    cell_seg_data_summary_file=summary,
                                    score_data_file=score,
                                    tissue_seg_data_summary_file=tissue_seg_data,
                                    verbose=verbose)
                image_id = str(uuid4())
                self.images[image_id] = cid
                rows.append([sample,frame,image_id])
        return pd.DataFrame(rows)     


class CellImageDataInForm(CellImageDataGeneric):
    """ Store data from a single image from an inForm export
    """
    def __init__(self):
        super().__init__()
    @property
    def excluded_channels(self):
        return ['Autofluorescence','Post-processing']    
    
    def read_image_data(self,
                        cell_seg_data_file=None,
                        cell_seg_data_summary_file=None,
                        score_data_file=None,
                        tissue_seg_data_summary_file=None,
                        verbose=False,
                        channel_abbreviations=None):
        """ Read in the image data from a inForm

        :param cell_seg_data_file:
        :type string:

        """
        _seg = pd.read_csv(cell_seg_data_file,"\t")
        if 'Tissue Category' not in _seg: _seg['Tissue Category'] = 'Any'

        ##########
        # Set the cells
        _cells = _seg.loc[:,['Cell ID','Cell X Position','Cell Y Position']].\
                              rename(columns={'Cell ID':'cell_index',
                                              'Cell X Position':'x',
                                              'Cell Y Position':'y'})
        _cells = _cells.applymap(int).set_index('cell_index')

        ###########
        # Set the cell phenotypes
        #    Try to read phenotypes from the summary file if its there because some may be zero counts and won't show up in the cell_seg file
        if 'Phenotype' in _seg:
            # Sometimes inform files won't have a Phenotype columns
            _phenotypes = _seg.loc[:,['Cell ID','Phenotype']]
        else:
            _phenotypes = _seg.loc[:,['Cell ID']]
            _phenotypes['Phenotype'] = np.nan
        _phenotypes = _phenotypes.rename(columns={'Cell ID':'cell_index','Phenotype':'phenotype_label'})
        _phenotypes_present = pd.Series(_phenotypes['phenotype_label'].unique()).tolist()
        if np.nan not in _phenotypes_present: _phenotypes_present = _phenotypes_present + [np.nan] 
        _phenotype_list = pd.DataFrame({'phenotype_label':_phenotypes_present})
        _phenotype_list.index.name = 'phenotype_index'
        _phenotype_list = _phenotype_list.reset_index()

        if cell_seg_data_summary_file is not None:
             ############
             # Update the phenotypes table if a cell_seg_data_summary file is present
            if verbose: sys.stderr.write("cell seg summary file is present so acquire phenotype list from it\n")
            _segsum = pd.read_csv(cell_seg_data_summary_file,"\t")
            if 'Phenotype' not in _segsum.columns: 
                if verbose: sys.stderr.write("missing phenotype column\n")
                _segsum['Phenotype'] = np.nan

            _phenotypes_present = [x for x in sorted(_segsum['Phenotype'].unique().tolist()) if x != 'All']
            if np.nan not in _phenotypes_present: _phenotypes_present = _phenotypes_present + [np.nan] 
            _phenotype_list = pd.DataFrame({'phenotype_label':_phenotypes_present})
            _phenotype_list.index.name = 'phenotype_index'
            _phenotype_list = _phenotype_list.reset_index()

        _phenotypes = _phenotypes.merge(_phenotype_list,on='phenotype_label')
        _phenotype_list = _phenotype_list.set_index('phenotype_index')
        #Assign 'phenotypes' in a way that ensure we retain the pre-defined column structure
        self.set_data('phenotypes',_phenotype_list)

        _phenotypes = _phenotypes.drop(columns=['phenotype_label']).applymap(int).set_index('cell_index')

        # Now we can add to cells our phenotype indecies
        _cells = _cells.merge(_phenotypes,left_index=True,right_index=True,how='left')


        ###########
        # Set the cell_regions
        if tissue_seg_data_summary_file is not None:
            raise ValueError("Region summary not implemented")
        else:
            _cell_regions = _seg[['Cell ID','Tissue Category']].copy().rename(columns={'Cell ID':'cell_index','Tissue Category':'region_label'})
            _regions = pd.DataFrame({'region_label':_cell_regions['region_label'].unique()})
            _regions.index.name = 'region_index'
            self.set_data('regions',_regions)
            _cell_regions = _cell_regions.merge(self.get_data('regions').reset_index(),on='region_label')
            _cell_regions = _cell_regions.drop(columns=['region_label']).set_index('cell_index')

        # Now we can add to cells our region indecies
        _cells = _cells.merge(_cell_regions,left_index=True,right_index=True,how='left')


        # Assign 'cells' in a way that ensures we retain our pre-defined column structure. Should throw a warning if anything is wrong
        self.set_data('cells',_cells)

        ###########
        # Get the intensity measurements - sets 'measurement_channels', 'measurement_statistics', 'measurement_features', and 'cell_measurements'
        self._parse_measurements(_seg,channel_abbreviations)  

        ###########
        # Get the thresholds
        if score_data_file is not None: 
            self._parse_score_file(score_data_file)
        return

    def _parse_measurements(self,_seg,channel_abbreviations):   
        # Parse the cell seg pandas we've already read in to get the cell-level measurements, as well as what features we are measuring
        # Sets the 'measurement_channels', 'measurement_statistics', 'measurement_features', and 'cell_measurements'
        keepers = ['Cell ID']

        # Some older versions don't have tissue category
        if 'Entire Cell Area (pixels)' in _seg.columns: keepers.append('Entire Cell Area (pixels)')

        keepers2 = [x for x in _seg.columns if re.search('Entire Cell.*\s+\S+ \(Normalized Counts, Total Weighting\)$',x)]
        keepers3 = [x for x in _seg.columns if re.search('\s+\S+ \(Normalized Counts, Total Weighting\)$',x) and x not in keepers2]
        _intensity1 = []
        for cname in keepers2:
            m = re.match('Entire Cell\s+(.*) (Mean|Min|Max|Std Dev|Total) \(Normalized Counts, Total Weighting\)$',cname)
            stain = m.group(1)
            v = _seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                _intensity1.append([row[0],stain,m.group(2),round(row[1],_float_decimals)])
        _intensity1 = pd.DataFrame(_intensity1,columns=['cell_index','channel_label','statistic_label','value'])
        _intensity1['feature_label'] = 'Whole Cell'

        _intensity2 = []
        _intensity3 = []
        for cname in keepers3:
            if re.match('Entire Cell',cname): continue
            m = re.match('(\S+)\s+(.*) (Mean|Min|Max|Std Dev|Total) \(Normalized Counts, Total Weighting\)$',cname)
            compartment = m.group(1)
            stain = m.group(2)
            v = _seg[['Cell ID',cname,compartment+' Area (pixels)']]
            v.columns = ['Cell ID','value','value1']
            v = v.copy()
            for row in v.itertuples(index=False):
                _intensity2.append([row[0],stain,compartment,m.group(3),round(row[1],_float_decimals)])
                _intensity3.append([row[0],'Post-processing',compartment,'Area (pixels)',round(row[2],_float_decimals)])

        _intensity2 = pd.DataFrame(_intensity2,columns=['cell_index','channel_label','feature_label','statistic_label','value'])
        _intensity3 = pd.DataFrame(_intensity3,columns=['cell_index','channel_label','feature_label','statistic_label','value'])

        _intensities = [_intensity2,_intensity3,_intensity1.loc[:,_intensity2.columns]]
        if 'Entire Cell Area (pixels)' in _seg:
            _intensity4 = _seg[['Cell ID','Entire Cell Area (pixels)']].rename(columns={'Cell ID':'cell_index',
                                                                                 'Entire Cell Area (pixels)':'value',
                                                                                })
            _intensity4['channel_label'] = 'Post-processing'
            _intensity4['feature_label'] = 'Whole Cell'
            _intensity4['statistic_label'] = 'Area (pixels)'
            _intensities += [_intensity4.loc[:,_intensity2.columns]]
        _intensity = pd.concat(_intensities)

        _measurement_channels = pd.DataFrame({'channel_label':_intensity['channel_label'].unique()})
        _measurement_channels.index.name = 'channel_index'
        _measurement_channels['channel_abbreviation'] = _measurement_channels['channel_label']
        if channel_abbreviations:
            _measurement_channels['channel_abbreviation'] = \
                _measurement_channels.apply(lambda x: x['channel_label'] if x['channel_label'] not in channel_abbreviations else channel_abbreviations[x['channel_label']],1)
        self.set_data('measurement_channels',_measurement_channels)

        _measurement_statistics = pd.DataFrame({'statistic_label':_intensity['statistic_label'].unique()})
        _measurement_statistics.index.name = 'statistic_index'
        self.set_data('measurement_statistics',_measurement_statistics)

        _measurement_features = pd.DataFrame({'feature_label':_intensity['feature_label'].unique()})
        _measurement_features.index.name = 'feature_index'
        self.set_data('measurement_features',_measurement_features)

        _cell_measurements = _intensity.merge(self.get_data('measurement_channels').reset_index(),on='channel_label',how='left').\
                          merge(self.get_data('measurement_statistics').reset_index(),on='statistic_label',how='left').\
                          merge(self.get_data('measurement_features').reset_index(),on='feature_label',how='left').\
                          drop(columns=['channel_label','feature_label','statistic_label','channel_abbreviation'])
        _cell_measurements.index.name = 'measurement_index'
        self.set_data('cell_measurements',_cell_measurements)


    def _parse_score_file(self,score_data_file):
        # Sets the 'thresholds' table by parsing the score file
        _score_data = pd.read_csv(score_data_file,"\t")
        if 'Tissue Category' not in _score_data:
            raise ValueError('cannot read Tissue Category from '+str(score_file))
        _score_data.loc[_score_data['Tissue Category'].isna(),'Tissue Category'] = 'Any'

        _score_data = _score_data[['Tissue Category','Cell Compartment','Stain Component','Positivity Threshold']].\
                      rename(columns={'Tissue Category':'region_label',
                                      'Cell Compartment':'feature_label',
                                      'Stain Component':'channel_label',
                                      'Positivity Threshold':'threshold_value'})
        _score_data.index.name = 'gate_index'
        _score_data = _score_data.reset_index('gate_index')
        # We only want to read the 'Mean' statistic for thresholding
        _mystats = self.get_data('measurement_statistics')
        _score_data['statistic_index'] = _mystats[_mystats['statistic_label']=='Mean'].iloc[0].name 
        _thresholds = _score_data.merge(self.get_data('measurement_features').reset_index(),on='feature_label').\
                                  merge(self.get_data('measurement_channels').reset_index(),on='channel_label').\
                                  merge(self.get_data('regions').reset_index(),on='region_label').\
                                  drop(columns=['feature_label','channel_label','region_label'])
        # By default for inform name the gate after the channel abbreviation
        _thresholds['gate_label'] = _thresholds['channel_abbreviation']
        _thresholds = _thresholds.drop(columns=['channel_abbreviation'])
        _thresholds = _thresholds.set_index('gate_index')
        self.set_data('thresholds',_thresholds)

class Frame:
    def __init__(self,path,mydir,sample,frame,seg_file,score_file,summary_file,binary_seg_maps,tissue_seg_data,verbose=False):
        if verbose: sys.stderr.write("FRAME: "+str(seg_file)+"\n")
        ### Read in the files here and create a dataframe
        self._seg = pd.read_csv(seg_file,"\t")
        self._scores = self._read_vectra_score_file(score_file)
        self._summary = None
        self._tissue_seg = None
        if tissue_seg_data is not None:
            self._tissue_seg = pd.read_csv(tissue_seg_data,sep='\t')

        ## check if Phenotype is missing, and set it to null if it is
        if 'Phenotype' not in self._seg:
            self._seg['Phenotype'] = 'unspecified'

        # get the enumeration of the components from a pattern match
        # get the stains and thresholds
        self._stains = OrderedDict() #read into a per-tissue structure
        for tissue in self._scores:
            if tissue not in self._stains:
                self._stains[tissue] = OrderedDict()
            # Look for multi-stain case will be like "First" Cell Compartment, "Second" Cell Compartment etc.
            checks = [re.match('(\S+)\s+',x).group(1) for x in list(self._scores[tissue].keys()) if re.match('\S+ Cell Compartment$',x)]
            for check in checks:
                compartment = check+' Cell Compartment'
                stain = check+' Stain Component'
                if compartment in self._scores[tissue]:
                    self._stains[tissue][self._scores[tissue][stain]] = OrderedDict({'compartment':self._scores[tissue][compartment],
                        'threshold':self._get_multi_threshold(tissue,self._scores[tissue][stain])})
            # if there were no values found in "checks" then it is probably a single stain score
            if len(checks) == 0:
                compartment = 'Cell Compartment'
                stain = 'Stain Component'
                if compartment in self._scores[tissue]:
                    self._stains[tissue][self._scores[tissue][stain]] = OrderedDict({'compartment':self._scores[tissue][compartment],
                        'threshold':self._get_single_threshold(tissue,self._scores[tissue][stain])})
        ### Finished reading in scores

        if summary_file:
            self._summary = pd.read_csv(summary_file,sep="\t")

        ##### FINISHED READING IN THINGS NOW OUTPUT THINGS ##########
        keepers = ['Cell ID','Phenotype',
            'Cell X Position',
            'Cell Y Position']

        # Some older versions don't have tissue category
        if 'Tissue Category' in self._seg.columns: keepers.append('Tissue Category')
        if 'Entire Cell Area (pixels)' in self._seg.columns: keepers.append('Entire Cell Area (pixels)')

        keepers2 = [x for x in self._seg.columns if re.search('Entire Cell.*\s+\S+ \(Normalized Counts, Total Weighting\)$',x)]
        keepers3 = [x for x in self._seg.columns if re.search('\s+\S+ \(Normalized Counts, Total Weighting\)$',x) and x not in keepers2]
        entire = OrderedDict()
        for cname in keepers2:
            m = re.match('Entire Cell\s+(.*) (Mean|Min|Max|Std Dev|Total) \(Normalized Counts, Total Weighting\)$',cname)
            stain = m.group(1)
            v = self._seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in entire: entire[row[0]] = OrderedDict()
                if stain not in entire[row[0]]: entire[row[0]][stain] = OrderedDict()
                entire[row[0]][stain][m.group(2)]=round(row[1],_float_decimals)
        compartment_areas = OrderedDict()
        compartments = OrderedDict()
        for cname in keepers3:
            if re.match('Entire Cell',cname): continue
            m = re.match('(\S+)\s+(.*) (Mean|Min|Max|Std Dev|Total) \(Normalized Counts, Total Weighting\)$',cname)
            compartment = m.group(1)
            stain = m.group(2)
            v = self._seg[['Cell ID',cname,compartment+' Area (pixels)']]
            v.columns = ['Cell ID','value','value1']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in compartments: compartments[row[0]] = OrderedDict()
                if stain not in compartments[row[0]]: compartments[row[0]][stain] = OrderedDict()
                if compartment not in compartments[row[0]][stain]: compartments[row[0]][stain][compartment] = OrderedDict()
                compartments[row[0]][stain][compartment][m.group(3)] = round(row[1],_float_decimals)
                #compartments[row[0]][stain][compartment]['Area'] = round(row[2],_float_decimals)
                if row[0] not in compartment_areas: compartment_areas[row[0]] = OrderedDict()
                compartment_areas[row[0]][compartment] = round(row[2],_float_decimals)
        v = self._seg[keepers].copy()
        if 'Entire Cell Area (pixels)' not in v.columns: v['Entire Cell Area (pixels)'] = np.nan #incase not set
        if 'Tissue Category' not in v.columns: v['Tissue Category'] = 'Any'
        v['compartment_areas'] = v.apply(lambda x: compartment_areas[x['Cell ID']],1)
        v['compartment_values'] = v.apply(lambda x: compartments[x['Cell ID']],1)
        v['entire_cell_values'] = v.apply(lambda x: np.nan if x['Cell ID'] not in entire else entire[x['Cell ID']],1) #sometimes not present
        #v = self._seg[keepers+keepers2]
        v = v.rename(columns = {'Cell ID':'id',
            'Entire Cell Area (pixels)':'cell_area',
            'Cell X Position':'x',
            'Cell Y Position':'y',
            'Phenotype':'phenotype',
            'Tissue Category':'tissue'})
        v['folder'] = mydir.lstrip('/').lstrip('\\')
        v['sample'] = sample
        v['frame'] = frame
        v['frame_stains'] = None
        v['frame_stains'] = v.apply(lambda x: json.dumps(self._stains),1) 


        if self._summary is not None:
            if 'Phenotype' not in self._summary.columns: 
                if verbose: sys.stderr.write("missing phenotype column\n")
                self._summary['Phenotype'] = 'unspecified'
            #### Read our areas from the summary #####
            if verbose: sys.stderr.write(str(self._get_vectra_frame_areas())+"\n")
            myareas = OrderedDict(self._get_vectra_frame_areas())
            ### Now we have our areas read lets put that data into things
            tissues_present = [x for x in myareas.keys() if x != 'All']
            myarea2 = OrderedDict()
            for tissue in tissues_present:
                myarea2[tissue] = int(myareas[tissue])


            # Use 'All' to read total area by default eventually can default to mask
            v['total_area'] = v.apply(lambda x: myareas['any'] if 'All' not in myareas else myareas['All'],1)
            v['tissues_present'] = v.apply(lambda x: json.dumps(myarea2),1)
            ### Lets the phenotypes that are here
            phenotypes_present = [x for x in sorted(self._summary['Phenotype'].unique().tolist()) if x != 'All']
            v['phenotypes_present'] = v.apply(lambda x: json.dumps(phenotypes_present),1)
        else:
            # We need to get these values from elsewhere
            sys.stderr.write("Guessing at phenotypes present\n")
            v['tissues_present'] = np.nan
            v['phenotypes_present'] = json.dumps(list(v['phenotype'][v['phenotype'].notna()].unique()))
        self._cells = v
    @property
    def cells (self):
        return self._cells
    def _get_vectra_frame_areas(self):
        # If we have tissue segmentation lets use that
        if self._tissue_seg is not None:
            df = self._tissue_seg[['Tissue Category','Region Area (pixels)']].rename(columns={'Tissue Category':'tissue','Region Area (pixels)':'tissue_area'})
            sum_area = np.sum(df['tissue_area'])
            df = df.append({'tissue':'All','tissue_area':sum_area},ignore_index=True)
            return df.set_index('tissue')['tissue_area'].to_dict()
            
        # At this point we would need a summary file present to get the areas
        if self._summary is None: raise ValueError("You need summary files present to get areas")
        df = self._summary.copy()
        mega = df.apply(lambda x: np.nan if float(x['Cell Density (per megapixel)']) == 0 else int(1000000*float(x['Total Cells'])/float(x['Cell Density (per megapixel)'])),1) # cell area in mega pixels
        df['Summary Area Megapixels'] = mega
        # in case phenotype wasn't specified
        if 'Phenotype' not in df.columns: df['Phenotype'] = 'unspecified'
        # some summary files don't have tissue category
        if 'Tissue Category' in df.columns:
            keepers = ['Tissue Category','Phenotype','Summary Area Megapixels']
        else:
            keepers = ['Phenotype','Summary Area Megapixels']
        df = df[keepers].copy()

        # Set our default tissue name to "any" if none exists
        if 'Tissue Category' not in df.columns: df['Tissue Category'] = 'any'
        df = df.rename(columns={'Tissue Category':'tissue',
                            'Phenotype':'phenotype',
                            'Summary Area Megapixels':'tissue_area'
                           })
        df = df[df['tissue_area'].notna()]
        if df[df['phenotype']=='All'].shape[0] > 0:
            return df.loc[(df['phenotype']=='All'),['tissue','tissue_area']].set_index('tissue')['tissue_area'].to_dict()
        return df.loc[(df['phenotype']=='unspecified'),['tissue','tissue_area']].set_index('tissue')['tissue_area'].to_dict()

    def _get_multi_threshold(self,tissue,stain):
        v = [x for x in self._scores[tissue].keys() if re.search(' Threshold$',x)]
        for entry in v:
            name = re.match('(.*) Threshold$',entry).group(1)
            if name == stain:
                return self._scores[tissue][entry]
        raise ValueError('did not find tissue and stain '+str(tissue)+' '+str(stain)+' '+str(self._scores))
    def _get_single_threshold(self,tissue,stain):
        return self._scores[tissue]['Positivity Threshold']

    def _read_vectra_score_file(self,score_file):
        # Read the score file into an Ordered Dictionary of tissues
        # This order dictionary contains series for each line of the score file
        _scores = OrderedDict()
        sfile = pd.read_csv(score_file,"\t")
        if 'Tissue Category' not in sfile:
            raise ValueError('cannot read Tissue Category from '+str(score_file))
        sfile.loc[sfile['Tissue Category'].isna(),'Tissue Category'] = 'any'
        head = sfile.columns
        obs_sample = None
        for row in sfile.itertuples(index=False):
            #row = row.to_dict()
            s = pd.Series(row,index=head)
            if obs_sample is not None and obs_sample != s['Sample Name']:
                raise ValueError('Multiple samples defined in one score file.  weird. '+score_file)
            obs_sample = s['Sample Name']
            if s['Tissue Category'] in _scores: raise ValueError('Same tissue is scored multiple times '+"\n"+str(s))
            _scores[s['Tissue Category']] = OrderedDict(s.to_dict())
        return _scores




class SampleSet:
    """ Read in a Folder containing sample folders recursively

    .. note:: This connection class is the primary portal to
              work with the REDCap system

    :param path: sample folder or folder containing sample folders
    :type path: string
    """
    def __init__(self,path,verbose=False,limit=None,sample_index=1):
        
        base = os.path.abspath(path)
        path = os.path.abspath(path)
        self._path = path
        self._cells = None
        self._samples = []
        z = 0
        for p, dirs, files in os.walk(self._path,followlinks=True,topdown=False):
            mydir = p[len(path):]
            z += 1
            segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
            if len(segs) == 0: continue
            if verbose: sys.stderr.write("SAMPLE: "+str(p)+"\n")
            s = Sample(p,mydir,verbose,sample_index)
            self._samples.append(s)
            if limit is not None and z >= limit: break
    @property
    def cells(self): 
        if self._cells is not None: return self._cells
        v = pd.concat([x.cells for x in self._samples])
        self._cells = v
        return self._cells

class Sample:
    def __init__(self,path,mydir,verbose=False,sample_index=1):
        self._path = path
        self._cells = None
        files = os.listdir(path)
        # Find frames in the same by a filename type we should have
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        sample_folder = path.split(os.sep)[-1*sample_index] #os.path.basename(path)
        self._frames = OrderedDict()
        snames = set()
        for file in segs:
            m = re.match('(.*)cell_seg_data.txt$',file)
            score = os.path.join(path,m.group(1)+'score_data.txt')
            summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
            binary_seg_maps = os.path.join(path,m.group(1)+'binary_seg_maps.tif')
            tfile = os.path.join(path,m.group(1)+'tissue_seg_data_summary.txt')
            tissue_seg_data = tfile if os.path.exists(tfile) else None
            sample = sample_folder
            snames.add(sample)
            frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if not os.path.exists(summary):
                if verbose: sys.stderr.write('Missing summary file '+summary+"\n")
                summary = None
            if not os.path.exists(score):
                raise ValueError('Missing score file '+score)
            self._frames[frame] = Frame(path,mydir,sample,frame,data,score,summary,binary_seg_maps,tissue_seg_data,verbose)
        if len(snames) > 1:
            raise ValueError('Error multiple samples in folder '+path)
        self._sample_name = list(snames)[0]        
    @property
    def cells(self):
        if self._cells is not None: return self._cells
        v =  pd.concat([x.cells for x in self._frames.values()])
        self._cells = v
        return(v)
