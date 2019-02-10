import pandas as pd
class Measurement(pd.DataFrame):
    """
    The parent class for classes to report measurements.

    """
    _metadata = ['microns_per_pixel','measured_regions','measured_phenotypes','cellframe'] # for extending dataframe to include this property
    def __init__(self,*args, **kw):
        super(Measurement, self).__init__(*args, **kw) 
    @property
    def _constructor(self):
        return Measurement
    @classmethod
    def read_cellframe(cls,cdf,measured_regions=None,measured_phenotypes=None,**kwargs):
        v = cls(cls._preprocess_dataframe(cdf,**kwargs))
        if measured_regions is None: v.measured_regions = cdf.get_measured_regions()
        if measured_phenotypes is None: v.measured_phenotypes = cdf.phenotypes
        v.microns_per_pixel = cdf.microns_per_pixel
        v.cellframe = cdf.copy()
        return v