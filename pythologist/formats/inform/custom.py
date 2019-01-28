from pythologist.formats.inform.frame import CellFrameInForm
from pythologist.formats.utilities import read_tiff_stack, make_binary_image_array, map_image_ids, watershed_image
from uuid import uuid4
import pandas as pd
import numpy as np


class CellFrameInFormLineArea(CellFrameInForm):
    def __init__(self):
        super().__init__()
        ### Define extra InForm-specific data tables
        self.data_tables['custom_images'] = {'index':'db_id',
                 'columns':['custom_label','image_id']}
        for x in self.data_tables.keys():
            if x in self._data: continue
            self._data[x] = pd.DataFrame(columns=self.data_tables[x]['columns'])
            self._data[x].index.name = self.data_tables[x]['index']
    def set_line_area(self,line_image,area_image, steps=20):
        #regions = prepare_margin_line_tumor_area(line_image,area_image)
        drawn_binary = read_tiff_stack(line_image)[0]['raw_image']
        drawn_binary = make_binary_image_array(drawn_binary)

        image_id = uuid4().hex
        self._images[image_id] = drawn_binary
        df = pd.DataFrame(pd.Series({'custom_label':'Drawn','image_id':image_id})).T
        df.index.name = 'db_id'
        self.set_data('custom_images',df)

        ids = map_image_ids(drawn_binary,remove_zero=False)
        zeros = ids.loc[ids['id']==0]
        zeros = list(zip(zeros['x'],zeros['y']))
        valid = ids.loc[ids['id']!=0]
        valid = list(zip(valid['x'],valid['y']))
        grown = drawn_binary.copy()
        area_binary = read_tiff_stack(area_image)[0]['raw_image']
        area_binary = make_binary_image_array(area_binary)

        ids = map_image_ids(drawn_binary,remove_zero=False)
        zeros = ids.loc[ids['id']==0]
        zeros = list(zip(zeros['x'],zeros['y']))
        valid = ids.loc[ids['id']!=0]
        valid = list(zip(valid['x'],valid['y']))
        grown = watershed_image(drawn_binary,valid,zeros,steps = steps).astype(np.int8)
        processed_image = self.get_image(self.processed_image_id).astype(np.int8)
        #return {'grown':grown,'tumor':area_binary,'processed':processed_image}
        margin_binary = grown&processed_image
        tumor_binary = (area_binary&(~grown))&processed_image
        stroma_binary = (~((tumor_binary|grown)&processed_image))&processed_image

        #regions - we will replace all regions
        d = {'Margin':margin_binary,
                          'Tumor':tumor_binary,
                          'Stroma':stroma_binary}
        self.set_regions(d)
        return d
