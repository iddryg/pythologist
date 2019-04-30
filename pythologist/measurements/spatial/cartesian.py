from pythologist.measurements import Measurement
import pandas as pd
import numpy as np
import math, sys
class Cartesian(Measurement):
    @staticmethod
    def _preprocess_dataframe(cdf,subsets,step_pixels,max_distance_pixels,*args,**kwargs):
        def _hex_coords(frame_shape,step_pixels):
            halfstep = int(step_pixels/2)
            vstep = int(step_pixels*0.85)
            coordinates = []
            for i,y in enumerate(range(0,frame_shape[0]+step_pixels,vstep)):
                #iterate over the x coords
                for x in range(halfstep if i%2==1 else 0,frame_shape[1]+step_pixels,step_pixels):
                    coordinates.append((x,y))
            return pd.DataFrame(coordinates,columns=['frame_x','frame_y'])
        frames = cdf.groupby(cdf.frame_columns).first()[['frame_shape']]
        frames['frame_coords'] = frames['frame_shape'].apply(lambda shape: _hex_coords(shape,step_pixels))

        allcoords = []
        for i,r in frames.iterrows():
            #if 'verbose' in kwargs and kwargs['verbose']: sys.stderr.write("Reading frame\n"+str(r)+"\n\n")
            idf = pd.DataFrame([i],columns=frames.index.names)
            idf['_key'] = 1
            coords = r['frame_coords']
            coords['_key'] = 1
            coords = idf.merge(coords,on='_key').drop(columns='_key')
            coords['frame_shape'] = 0
            coords['frame_shape'] = coords['frame_shape'].apply(lambda x: r['frame_shape'])
            allcoords.append(coords)
        allcoords = pd.concat(allcoords)
        allcoords['step_pixels'] = step_pixels
        allcoords = allcoords.reset_index(drop=True)
        allcoords.index.name = 'coord_id'

        ### Capture distances
        full = []
        for frame_id in allcoords['frame_id'].unique():
            fcdf = cdf.loc[cdf['frame_id']==frame_id]
            fcdf = fcdf.dropna(subset=['phenotype_label'])
            primary = fcdf.copy()
            phenotypes = cdf.phenotypes
            # get the frame's CellDataFrame
            if subsets is not None:
                phenotypes = []
                subs = []
                for subset_logic in subsets:
                    sub = fcdf.subset(subset_logic,update=True)
                    subs.append(sub)
                    phenotypes.append(subset_logic.label)
                fcdf = pd.concat(subs)
            # get the frame's hex coordinates
            coords = allcoords.loc[allcoords['frame_id']==frame_id].copy().reset_index()
            counts = _get_proximal_points(fcdf,coords,
                                          fcdf.frame_columns,
                                          phenotypes,
                                          max_distance_pixels)
            totals = _get_proximal_points(primary,coords,
                                          primary.frame_columns,
                                          cdf.phenotypes,
                                          max_distance_pixels)
            totals = totals.groupby(cdf.frame_columns+['coord_id','frame_x','frame_y']).sum()[['count']].rename(columns={'count':'total'}).\
                reset_index()
            counts = counts.merge(totals,on=cdf.frame_columns+['coord_id','frame_x','frame_y'])
            counts['count'] = counts['count'].astype(int)
            counts['total'] = counts['total'].astype(int)
            counts['fraction'] = counts.apply(lambda x: np.nan if x['total']==0 else x['count']/x['total'],1)
            full.append(counts)
        full = pd.concat(full).reset_index(drop=True)
        full['max_distance_pixels'] = max_distance_pixels
        return full

    def rgb_dataframe(self,max_quantile_color=0.95,red=None,green=None,blue=None):
        df2 = self.copy()
        d1 = df2.groupby(['frame_id','phenotype_label']).\
            min()[['fraction']].reset_index().rename(columns={'fraction':'minimum'})
        d1['minimum'] = 0
        d2 = df2.groupby(['frame_id','phenotype_label']).\
            max()[['fraction']].reset_index().rename(columns={'fraction':'maximum'})
        d3 = df2.groupby(['frame_id','phenotype_label']).\
            quantile(max_quantile_color)[['fraction']].reset_index().rename(columns={'fraction':'p95'})
        df2 = d1.merge(d2,on=['frame_id','phenotype_label'],how='outer').merge(df2,on=['frame_id','phenotype_label'],how='outer').\
            merge(d3,on=['frame_id','phenotype_label'],how='outer')
        df2 = df2.fillna(0)
        df2['maximum'] = df2['p95']
        df2['fraction'] = df2.apply(lambda x: x['fraction'] if x['fraction'] < x['p95'] else x['p95'],1)
        df2['range'] = df2['maximum'].subtract(df2['minimum'])
        df2.loc[df2['range']<=0,'range'] =1
        df2['standardized'] = (df2['fraction'].subtract(df2['minimum'])).divide(df2['range']).multiply(255).astype(int)
        rangetop = df2[self.cdf.frame_columns+['phenotype_label','p95']].drop_duplicates().\
            rename(columns={'p95':'range_top'})
        df3 = df2.set_index(self.cdf.frame_columns+['coord_id','frame_x','frame_y','frame_shape','step_pixels'])[['phenotype_label','standardized']].\
            pivot(columns='phenotype_label')
        df3.columns = df3.columns.droplevel(0)
        df3 = df3.reset_index()
        df3['zero'] = 0
        #return df3
        if red is None: red = 'zero'
        if green is None: green = 'zero'
        if blue is None: blue = 'zero'
        df3['color'] = df3.apply(lambda x: (x[red],x[green],x[blue]),1)
        df3['color_str'] = df3.apply(lambda x: '#%02x%02x%02x' % x['color'],1).astype(str)
        df3 = df3.sort_values(['frame_id','frame_y','frame_x']).reset_index(drop=True)
        dcols = df3[['color','color_str']].drop_duplicates()
        df3['color_str'] = pd.Categorical(df3['color_str'],categories=dcols['color_str'])
        return df3, dcols['color_str'], rangetop
        
def _get_proximal_points(cdf_lines,coords,frame_columns,phenotype_labels,max_distance_pixels):
    prox = cdf_lines.merge(coords,on=frame_columns)
    prox['x1'] = prox['x'].subtract(prox['frame_x'])
    prox['y1'] = prox['y'].subtract(prox['frame_y'])
    prox['x2'] = prox['x1'].multiply(prox['x1'])
    prox['y2'] = prox['y1'].multiply(prox['y1'])
    prox['x2y2'] = prox['x2']+prox['y2'] 
    prox['distance'] = prox['x2y2'].apply(math.sqrt)
    prox = prox.drop(columns=['x1','x2','y1','y2','x2y2'])
    prox = prox.loc[prox['distance']<=max_distance_pixels]
    prox = prox.groupby(frame_columns+['coord_id','frame_x','frame_y','phenotype_label']).\
        count()[['cell_index']].rename(columns={'cell_index':'count'}).reset_index()
    coords['_key'] = 1
    phenos = pd.DataFrame({'phenotype_label':phenotype_labels})
    phenos['_key'] = 1
    cp = coords.merge(phenos,on='_key').drop(columns='_key')
    counts = cp.merge(prox,on=frame_columns+\
        ['coord_id','frame_x','frame_y','phenotype_label'],how='left').fillna(0)
    return counts