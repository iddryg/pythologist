import pandas as pd
from pythologist.measurements import Measurement
class Interface(Measurement):
    def __init__(self,cdf):
        self.cdf = cdf
        # narrow to the samples and frames
        self.images = pd.DataFrame(self.cdf.loc[:,self.cdf.frame_columns]).drop_duplicates()
    @staticmethod
    def _preprocess_dataframe(cdf,*args,**kwargs):
        return cdf

    def get_edge_maps(self):
        samples = self.images['sample_id'].unique().tolist()
        for sample_id in samples:
            s = self.cdf.db.get_sample(sample_id)
            for frame_id in self.images.loc[self.images['sample_id']==sample_id,'frame_id']:
                f = s.get_frame(frame_id)
                return f

