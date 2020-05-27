# pythologist-test-images

Images for use in unit tests etc.

List available datasets:

```python
from pythologist_test_images import TestImages

print(TestImages().datasets)
```

> ['IrisSpatialFeatures', 'Small', 'Tiny']

get a path pointing to the dataset

```python
path = TestImages().raw('IrisSpatialFeatures')
print(path)
```
> '/Source/pythologist-test-images/data/IrisSpatialFeatures/Example'

get a CellDataFrame pythologist object (requires pythologistreader and pythologist `pip install pythologist-test-images[h5]`)

```python
cdf = TestImages().celldataframe('Tiny')
tdf = cdf.threshold('T-CELL','PD1').threshold('OTHER','PDL1').threshold('TUMOR','PDL1')
tdf.loc[tdf['sample_name'].str.contains('I'),'project_name'] = 'SlideModelInfiltrated'
tdf.loc[tdf['sample_name'].str.contains('E'),'project_name'] = 'SlideModelExcluded'
tdf.loc[tdf['sample_name'].str.contains('U'),'project_name'] = 'SlideModelUniform'
tdf['project_id'] = tdf['project_name']
(ggplot(cnts,aes(x='sample_name',y='mean_density_mm2',
                 ymin='mean_density_mm2+stderr_density_mm2',
                 ymax='mean_density_mm2-stderr_density_mm2',fill='project_name'))
 + geom_bar(stat='identity')
 + geom_errorbar()
 + facet_wrap('phenotype_label',scales='free_y')
 + theme_bw()
 + theme(panel_spacing_x=0.5)
 + theme(figure_size=(6,6))
)
```

![Image of counts](https://github.com/jason-weirather/pythologist-test-images/raw/master/images/counts.png?raw=true)
