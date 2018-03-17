import os, re, json, sys
from collections import OrderedDict
import pandas as pd
import numpy as np


class MantraFrame:
    def __init__(self,path,mydir,sample,frame,seg_file,score_file,summary_file):
        self._scores = pd.read_csv(score_file,sep="\t")
        self._seg = pd.read_csv(seg_file,"\t")     
        self._stains = OrderedDict()
        #### Read in the files here and create a dataframe
        #### Read the score file first
        if self._scores['Sample Name'].unique().shape[0] > 1:
            raise ValueError('Multiple samples present in score file '+score_file)
        self._scores = self._scores[['Tissue Category',
                                     'Cell Compartment',
                                     'Stain Component',
                                     'Positivity Threshold']].drop_duplicates()
        # nan is too much trouble to deal with
        self._scores.loc[self._scores['Tissue Category'].isna(),'Tissue Category'] = ''
        for row in self._scores.itertuples(index=False):
            s = pd.Series(row,index=self._scores.columns)
            tissue = s['Tissue Category']
            compartment = s['Cell Compartment']
            threshold = s['Positivity Threshold']
            stain = s['Stain Component']
            if tissue not in self._stains: self._stains[tissue] = OrderedDict()
            if stain in self._stains[tissue]: raise ValueError('saw same stain twice '+score_file)
            self._stains[tissue][stain] = OrderedDict({
                'compartment':compartment,
                'threshold':threshold
            })
        #### Read the summary file if its there
        if summary_file:
            self._summary = pd.read_csv(summary_file,sep="\t")
        #### Now start in on the segmentation file
        keepers = ['Cell ID','Phenotype',
            'Cell X Position',
            'Cell Y Position',
            'Entire Cell Area (pixels)']
        keepers2 = [x for x in self._seg.columns if re.search('Entire Cell.*Mean \(Normalized Counts, Total Weighting\)$',x)]
        keepers3 = [x for x in self._seg.columns if re.search('Mean \(Normalized Counts, Total Weighting\)$',x) and x not in keepers2]
        entire = OrderedDict()
        for cname in keepers2:
            m = re.match('Entire Cell\s+(.*) Mean \(Normalized Counts, Total Weighting\)$',cname)
            stain = m.group(1)
            v = self._seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in entire: entire[row[0]] = OrderedDict()
                entire[row[0]][stain]=row[1]
        compartments = OrderedDict()
        for cname in keepers3:
            if re.match('Entire Cell',cname): continue
            m = re.match('(\S+)\s+(.*) Mean \(Normalized Counts, Total Weighting\)$',cname)
            compartment = m.group(1)
            stain = m.group(2)
            v = self._seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in compartments: compartments[row[0]] = OrderedDict()
                if stain not in compartments[row[0]]: compartments[row[0]][stain] = OrderedDict()
                compartments[row[0]][stain][compartment] = row[1]
        v = self._seg[keepers].copy()
        v['compartment_values'] = v.apply(lambda x: compartments[x['Cell ID']],1)
        v['entire_cell_values'] = v.apply(lambda x: entire[x['Cell ID']],1)
        #v = self._seg[keepers+keepers2]
        v = v.rename(columns = {'Cell ID':'id',
            'Entire Cell Area (pixels)':'cell_area',
            'Cell X Position':'x',
            'Cell Y Position':'y',
            'Phenotype':'phenotype',
            'Tissue Category':'tissue'})
        v['frame'] = frame
        v['sample'] = sample
        #v['threshold_marker'] = np.nan
        #v['threshold_call'] = np.nan
        #v['full_phenotype'] = v['phenotype']
        v['frame_stains'] = None
        v['frame_stains'] = v.apply(lambda x: json.dumps(self._stains),1) 
        v['tissue'] = ''
        if self._summary is not None:
            #### Read our areas from the summary #####
            myareas = OrderedDict(self._get_mantra_frame_areas())
            ## Now we have our areas read lets put that data into things
            tissues_present = [x for x in myareas.keys() if x != 'All']
            v['tissue_area'] = v.apply(lambda x: myareas[x['tissue']],1)
            v['total_area'] = v.apply(lambda x: myareas[''],1)
            v['tissues_present'] = v.apply(lambda x: json.dumps(tissues_present),1)
            ## Lets the phenotypes that are here
            phenotypes_present = [x for x in sorted(self._summary['Phenotype'].unique().tolist()) if x != 'All']
            v['phenotypes_present'] = v.apply(lambda x: json.dumps(phenotypes_present),1)
        else:
            v['tissue_area'] = np.nan
            v['total_area'] = np.nan
            v['phenotypes_present'] = json.dumps([])
            v['tissues_present'] = json.dumps({})
        v['folder'] = mydir.lstrip('/').lstrip('\\')
        self._cells = v
    def _get_mantra_frame_areas(self):
        if self._summary is None: raise ValueError("You need summary files present to get areas")
        df = self._summary.copy()
        mega = df.apply(lambda x: np.nan if float(x['Cell Density (per megapixel)']) == 0 else float(x['Total Cells'])/float(x['Cell Density (per megapixel)']),1) # cell area in mega pixels
        df['Summary Area Megapixels'] = mega
        # Lets ignore the cell specific things here
        #return(df[df['Phenotype']=='All'])
        df = df[['Phenotype','Summary Area Megapixels']].\
            rename(columns={'Phenotype':'phenotype',
                            'Summary Area Megapixels':'tissue_area'
                           })
        df['tissue'] = ''
        df = df.loc[df['phenotype']=='All',['tissue','tissue_area']].\
            set_index('tissue')['tissue_area'].to_dict()
        return df
    @property
    def cells (self):
        return self._cells



class VectraFrame:
    def __init__(self,path,mydir,sample,frame,seg_file,score_file,summary_file):
        ### Read in the files here and create a dataframe
        self._seg = pd.read_csv(seg_file,"\t")
        self._scores = self._read_vectra_score_file(score_file)
        self._summary = None
        # get the enumeration of the components from a pattern match
        # get the stains and thresholds
        self._stains = OrderedDict() #read into a per-tissue structure
        for tissue in self._scores:
            if tissue not in self._stains:
                self._stains[tissue] = OrderedDict()
            checks = [re.match('(\S+)\s+',x).group(1) for x in list(self._scores[tissue].keys()) if re.match('\S+ Cell Compartment$',x)]
            for check in checks:
                compartment = check+' Cell Compartment'
                stain = check+' Stain Component'
                if compartment in self._scores[tissue]:
                    self._stains[tissue][self._scores[tissue][stain]] = OrderedDict({'compartment':self._scores[tissue][compartment],
                        'threshold':self._get_vectra_threshold(tissue,self._scores[tissue][stain])})
        if summary_file:
            self._summary = pd.read_csv(summary_file,sep="\t")

        ##### FINISHED READING IN THINGS NOW OUTPUT THINGS ##########
        keepers = ['Cell ID','Phenotype',
            'Cell X Position',
            'Cell Y Position',
            'Entire Cell Area (pixels)','Tissue Category']
        keepers2 = [x for x in self._seg.columns if re.search('Entire Cell.*Mean \(Normalized Counts, Total Weighting\)$',x)]
        keepers3 = [x for x in self._seg.columns if re.search('Mean \(Normalized Counts, Total Weighting\)$',x) and x not in keepers2]
        entire = OrderedDict()
        for cname in keepers2:
            m = re.match('Entire Cell\s+(.*) Mean \(Normalized Counts, Total Weighting\)$',cname)
            stain = m.group(1)
            v = self._seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in entire: entire[row[0]] = OrderedDict()
                entire[row[0]][stain]=row[1]
        compartments = OrderedDict()
        for cname in keepers3:
            if re.match('Entire Cell',cname): continue
            m = re.match('(\S+)\s+(.*) Mean \(Normalized Counts, Total Weighting\)$',cname)
            compartment = m.group(1)
            stain = m.group(2)
            v = self._seg[['Cell ID',cname]]
            v.columns = ['Cell ID','value']
            v = v.copy()
            for row in v.itertuples(index=False):
                if row[0] not in compartments: compartments[row[0]] = OrderedDict()
                if stain not in compartments[row[0]]: compartments[row[0]][stain] = OrderedDict()
                compartments[row[0]][stain][compartment] = row[1]
        v = self._seg[keepers].copy()
        v['compartment_values'] = v.apply(lambda x: compartments[x['Cell ID']],1)
        v['entire_cell_values'] = v.apply(lambda x: entire[x['Cell ID']],1)
        #v = self._seg[keepers+keepers2]
        v = v.rename(columns = {'Cell ID':'id',
            'Entire Cell Area (pixels)':'cell_area',
            'Cell X Position':'x',
            'Cell Y Position':'y',
            'Phenotype':'phenotype',
            'Tissue Category':'tissue'})
        v['frame'] = frame
        v['sample'] = sample
        #v['threshold_marker'] = np.nan
        #v['threshold_call'] = np.nan
        #v['full_phenotype'] = v['phenotype']
        v['frame_stains'] = None
        v['frame_stains'] = v.apply(lambda x: json.dumps(self._stains),1) 


        if self._summary is not None:
            #### Read our areas from the summary #####
            
            myareas = OrderedDict(self._get_vectra_frame_areas())
            ### Now we have our areas read lets put that data into things
            tissues_present = [x for x in myareas.keys() if x != 'All']
            v['tissue_area'] = v.apply(lambda x: myareas[x['tissue']],1)
            v['total_area'] = v.apply(lambda x: myareas['All'],1)
            v['tissues_present'] = v.apply(lambda x: json.dumps(tissues_present),1)
            ### Lets the phenotypes that are here
            phenotypes_present = [x for x in sorted(self._summary['Phenotype'].unique().tolist()) if x != 'All']
            v['phenotypes_present'] = v.apply(lambda x: json.dumps(phenotypes_present),1)
        else:
            v['tissue_area'] = np.nan
            v['total_area'] = np.nan
            v['phenotypes_present'] = json.dumps([])
            v['tissues_present'] = json.dumps({})
        v['folder'] = mydir.lstrip('/').lstrip('\\')
        self._cells = v
    @property
    def cells (self):
        return self._cells
    def _get_vectra_frame_areas(self):
        if self._summary is None: raise ValueError("You need summary files present to get areas")
        df = self._summary.copy()
        mega = df.apply(lambda x: np.nan if float(x['Cell Density (per megapixel)']) == 0 else float(x['Total Cells'])/float(x['Cell Density (per megapixel)']),1) # cell area in mega pixels
        df['Summary Area Megapixels'] = mega
        # Lets ignore the cell specific things here
        #return(df[df['Phenotype']=='All'])
        df = df[['Tissue Category','Phenotype','Summary Area Megapixels']].\
            rename(columns={'Tissue Category':'tissue',
                            'Phenotype':'phenotype',
                            'Summary Area Megapixels':'tissue_area'
                           })
        df = df.loc[df['phenotype']=='All',['tissue','tissue_area']].set_index('tissue')['tissue_area'].to_dict()
        return df
    def _get_vectra_threshold(self,tissue,stain):
        v = [x for x in self._scores[tissue].keys() if re.search(' Threshold$',x)]
        for entry in v:
            name = re.match('(.*) Threshold$',entry).group(1)
            if name == stain:
                return self._scores[tissue][entry]
        raise ValueError('did not find tissue and stain '+str(tissue)+' '+str(stain)+' '+str(self._scores))

    def _read_vectra_score_file(self,score_file):
        # Read the score file into an Ordered Dictionary of tissues
        # This order dictionary contains series for each line of the score file
        _scores = OrderedDict()
        sfile = pd.read_csv(score_file,"\t")
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
    def __init__(self,path,verbose=False,limit=None,type="Vectra"):
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
            s = Sample(p,mydir,verbose,type=type)
            self._samples.append(s)
            if limit is not None and z >= limit: break
    @property
    def cells(self): 
        if self._cells is not None: return self._cells
        v = pd.concat([x.cells for x in self._samples])
        self._cells = v
        return self._cells

class Sample:
    def __init__(self,path,mydir,verbose=False,type="Vectra"):
        self._path = path
        self._cells = None
        files = os.listdir(path)
        # Find frames in the same by a filename type we should have
        segs = [x for x in files if re.search('_cell_seg_data.txt$',x)]
        sample_folder = os.path.basename(path)
        self._frames = OrderedDict()
        snames = set()
        for file in segs:
            m = re.match('(.*)_(\[\d+,\d+\])_cell_seg_data.txt$',file)
            #if not m: m = re.match('(.*)_([^_]+)_cell_seg_data.txt$',file)
            if m:
                sample = m.group(1)
                snames.add(sample)
                frame = m.group(2)
                summary = os.path.join(path,sample+'_'+frame+'_cell_seg_data_summary.txt')
                score = os.path.join(path,sample+'_'+frame+'_score_data.txt')

            else:
                # use folder as sample and file prefix as frame
                m = re.match('(.*)cell_seg_data.txt$',file)
                score = os.path.join(path,m.group(1)+'score_data.txt')
                summary = os.path.join(path,m.group(1)+'cell_seg_data_summary.txt')
                sample = sample_folder
                snames.add(sample)
                frame = m.group(1).rstrip('_')
            data = os.path.join(path,file)
            if not os.path.exists(summary):
                if verbose: sys.stderr.write('Missing summary file '+summary+"\n")
                summary = None
            if not os.path.exists(score):
                raise ValueError('Missing score file '+score)
            f = None
            if type == "Vectra":
                f = VectraFrame(path,mydir,sample,frame,data,score,summary)
            elif type == "Mantra":
                f = MantraFrame(path,mydir,sample,frame,data,score,summary)
            else: raise ValueError("Cannot read unsupported type "+type)
            self._frames[frame] = f
        if len(snames) > 1:
            raise ValueError('Error multiple samples in folder '+path)
        self._sample_name = list(snames)[0]        
    @property
    def cells(self):
        if self._cells is not None: return self._cells
        v =  pd.concat([x.cells for x in self._frames.values()])
        self._cells = v
        return(v)