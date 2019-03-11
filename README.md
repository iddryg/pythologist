# pythologist

*Read and analyze cell image data.*

Pythologist 1) reads exports from InForm software or other sources into a common storage format, and 2) extracts basic analysis features from cell image data.  This software is generally intended to be run from a jupyter notebook and provides hooks into the image data so that the user can have the flexability to execute analyses they design or find in the primary literature.

[List of large-scale image analysis publications](https://github.com/jason-weirather/pythologist/wiki/list-of-large-scale-image-analysis-publications)


Pythologist is based on [**IrisSpatialFeatures**](https://github.com/gusef/IrisSpatialFeatures) (C.D. Carey, ,D. Gusenleitner, M. Lipshitz, et al. Blood. 2017) https://doi.org/10.1182/blood-2017-03-770719, and is implemented in the python programming language. 

Features Pythologist add are:

* An common CellProjectGeneric storage class, and classical inheritance conventions to organize the importation of different data types.
* A mutable CellDataFrame class that can be used for slicing, and combining projects.
* The ability to add binary features to cells based on cell-cell contacts or cell proximity.
* Customizable images based on the cell segmentation or heatmaps spaninng the cartesian coordinates.
* Specify cell populations through a SubsetLogic syntax for quick selection of mutually exclusive phenotypes or binary features
* A set of Quality Check functions to identify potential issues in imported data.

# Installation

## Docker jupyter labs quickstart

To start a jupyter notebook in the current working directory on port 8885 you can use the following docker command.

**First build a docker image that will use your own user/group name and id.**

```
$ docker build -t pythologist:latest --build-arg user=$USERNAME \
                                     --build-arg group=$GROUPNAME \
                                     --build-arg user_id=$USERID \
                                     --build-arg group_id=$GROUPID .
```

**Now start the docker image.**

```
$ docker run --rm -p 8885:8888 -v $(pwd):/home/$USERNAME/work pythologist:latest
```

where `$USERNAME`, `$GROUPNAME`, `$USERID`, `$GROUPID` correspond to your user/group name/id.

## Install by pip

```
$ pip install pythologist
```

# Common tasks

### Reading in a project composed of InForm exports

#### Basic

The assumption here is that the exports are grouped so that sample folders contain one or more image exports, and that sample name can be inferred from the last folder name.

```python
from pythologist_test_images import TestImages
from pythologist_reader.formats.inform.sets import CellProjectInForm

# Get the path of the test dataset
path = TestImages().raw('IrisSpatialFeatures')

# Create the storage opbject where the project will be saved
cpi = CellProjectInForm('pythologist.h5',mode='w')

# Read the project data
cpi.read_path(path,project_name='IrisSpatialFeatures',
                   name_index=-1,
                   verbose='True',
                   microns_per_pixel=0.496)

# Create a cell dataframe for downstream analysis
cdf = cpi.cdf
```

#### Custom region annotations from tumor and invasive margin image drawings

Another format supported for a project import is one with a custom tumor and invasive margin definition.  Similar to above, the project is organized into sample folders, and each image within each sample folder has a *tif* file defining the tumor and invasive margin.  These come in the form of a `<image name prefix>_Tumor.tif` and `<image name prefix>_Invasive_Margin.tif` for each image.  The `_Tumor.tif` is an area filled in where the tumor is, and transparent elsewhere.  The `_Invasive_Margin.tif` is a drawn line of a known width.  `steps` is used to grow the margin out that many pixels in each direction to establish an invasive margin region.  Here we also rename some markers during read-in to clean up the syntax of thresholding on binary features.

```python
from pythologist_test_images import TestImages
from pythologist_reader.formats.inform.custom import CellProjectInFormLineArea

# Get the path of the test dataset
path = TestImages().raw('IrisSpatialFeatures')

# Specify where the data read-in will be stored as an h5 object
cpi = CellProjectInFormLineArea('pythologist.h5',mode='w')

# Read in the data (gets stored on the fly into the h5 object)
cpi.read_path(path,
              sample_name_index=-1,
              require=False,
              verbose=True,
              steps=76,
              project_name='IrisSpatialFeatures',
              channel_abbreviations={'PD-1 (Opal 540)':'PD1','PD-Ligand-1 (Opal 690)':'PDL1'},
              microns_per_pixel=0.496)

# Create a cell dataframe for downstream analysis
cdf = cpi.cdf

# Write out the cell dataframe
cdf.to_hdf('pythologist.cdf.h5','data')
``` 

*Note: we need to swap in an optimized watershed algorithm to speed up all these read operations.*

### Quality check samples

Check general status of the CellDataFrame

```python
import pythologist

cdf.qc(verbose=True).print_results()
```

View histograms of pixel intensity and the scoring of binary markers on each image

```python
qc = cdf.db.qc(verbose=True)
hist = qc.channel_histograms(0,20,100)
(ggplot(hist[hist['channel_label']=='CD86 (Opal 540)'],
        aes(x='bins',y='counts',fill='sample_name'))
 + geom_bar(stat='identity')
 + geom_vline(aes(xintercept='threshold_value',color='feature_label'))
 + facet_wrap(['sample_name','frame_name'])
 + theme_bw()
 + theme(figure_size=(20,20))
)
```

### Merge CellDataFrames that have the same image segmentations but different scored calls

This happens frequently because current InForm exports only permit two features to be scored per export

```python
merged,fail = cdf1.merge_scores(cdf2,on=['sample_name','frame_name','x','y'])
```


### Show names of the binary 'scored_calls'
```python
cdf.scored_names
```

### Combine two or more phenotypes into one or rename a phenotype
```python
collapsed = cdf.collapse_phenotypes(['CD68 PDL1+','CD68 PDL1-'],'CD68')
```

### Rename a tisssue

Rename *TUMOR* to *Tumor*

```python
renamed = cdf.rename_region('TUMOR','Tumor')
```

### Rename scored phenotypes

```python
renamed = cdf.rename_scored_calls({'PDL1 (Opal 520)':'PDL1'})
```

### Threshold a phenotype

Make *CYTOK* into *CYTOK PDL1+* and *CYTOK PDL1-*

```python
raw_thresh = raw.threshold('CYTOK','PDL1')
```

### Double threshold

```python
CD68_CD163 = raw.threshold('CD68','CD163').\
    threshold('CD68 CD163+','PDL1').\
    threshold('CD68 CD163-','PDL1')
```

### Get per frame counts

generate counts and fractions of the current phenotypes and export them to a csv

```python
cdf.counts().frame_counts().to_csv('my_frame_counts.csv')
```

### Get per sample counts

generate counts and fractions of the current phenotypes and export them to a csv

```python
cdf.counts().sample_counts().to_csv('my_sample_counts.csv')
```

### Identify cells that are in contact with a phenotype

The follow command creates a new CellDataFrame that has an additional binary feature representative of the contact with 'T cell' phenotype cells.

```python
cdf = cdf.contacts().threshold('T cell')
```

### Identify cells that are within some proximity of a phenotype of interest

The follow command creates a new CellDataFrame that has an additional binary feature representative of being inside or outisde 75 microns of 'T cell' phenotype cells.

```python
cdf = cdf.nearestneighbors().threshold('T cell','T cell/within 75um',distance_um=75)
```

### Create a cartesian map heatmap to show features

```python
df = sub.cartesian(subsets=[SL(phenotypes=['HRS+'],label='HRS+'),
                            SL(scored_calls={'CD86':'+'},label='CD86+'),
                            SL(phenotypes=['CTLA4+'],label='CTLA4+')],
                   step_pixels=75,max_distance_pixels=100,verbose=True)
fshape = df.iloc[0]['frame_shape']
rgbdf, col_list = df.rgb_dataframe(blue='CD86+')
line_color = '#444444'
(ggplot(rgbdf,aes(x='frame_x',y='frame_y',fill='color_str'))
 + geom_point(shape='h',stroke=0.6,size=10.2,color=line_color)
 + geom_hline(yintercept=0,color='gray')
 + geom_hline(yintercept=fshape[0]-1,color=line_color)
 + geom_vline(xintercept=0,color='gray')
 + geom_vline(xintercept=fshape[1]-1,color=line_color)
 + theme_bw()
 + theme(figure_size=(12,12),aspect_ratio=fshape[0]/fshape[1])
 + scale_fill_manual(col_list,guide=False)
 + scale_color_gradient(low='#555555',high='#5555FF')
 + scale_y_reverse()
)
```

### Create an image of cell-cell contacts between features of interest

```python

```

# Comparison to IrisSpatialFeatures

