# pythologist

*Read and analyze cell image data.*

## Intro

Pythologist 1) reads exports from InForm software or other sources into a common storage format, and 2) extracts basic analysis features from cell image data.  This software is generally intended to be run from a jupyter notebook and provides hooks into the image data so that the user can have the flexability to execute analyses they design or find in the primary literature.

[List of image analysis publications](https://github.com/jason-weirather/pythologist/wiki/list-of-image-analysis-publications)


Pythologist is based on [**IrisSpatialFeatures**](https://github.com/gusef/IrisSpatialFeatures) (C.D. Carey, ,D. Gusenleitner, M. Lipshitz, et al. Blood. 2017) https://doi.org/10.1182/blood-2017-03-770719, and is implemented in the python programming language. 

Features Pythologist add are:

* An common CellProjectGeneric storage class, and classical inheritance conventions to organize the importation of different data types.
* A mutable CellDataFrame class that can be used for slicing, and combining projects.
* The ability to add binary features to cells based on cell-cell contacts or cell proximity.
* Customizable images based on the cell segmentation or heatmaps spaninng the cartesian coordinates.
* Specify cell populations through a SubsetLogic syntax for quick selection of mutually exclusive phenotypes or binary features
* A set of Quality Check functions to identify potential issues in imported data.

## Documentation

Primary Software

* `pythologist` This software package uses a CellDataFrame class, an extension of a Pandas DataFrame to modify data and execute analyses [[Read the Docs](https://jason-weirather.github.io/pythologist/#modules)] [[source](https://github.com/jason-weirather/pythologist)]
    * `pythologist-schemas` This submodule documents/defines the formats of inputs and outputs expected in this pipeline. [[source](https://github.com/jason-weirather/pythologist-schemas)]
    * `pythologist-reader` This submodule facillitates reading platform-specific data into a harmonized format. [[Read the Docs](https://jason-weirather.github.io/pythologist-reader/)] [[source](https://github.com/jason-weirather/pythologist-reader)]
    * `pythologist-test-images` This submodule has some example data [[source](https://github.com/jason-weirather/pythologist-test-images)]
    * `pythologist-image-utilities` This submodule has helper functions to work with images [[Read the Docs](https://jason-weirather.github.io/pythologist-image-utilities/)] [[source](https://github.com/jason-weirather/pythologist-image-utilities)] 

Additional Analytics

* `good-neighbors` This package facilitates the analysis of cellular data based on their proximal "cellular neighborhoods" [[Read the Docs](https://jason-weirather.github.io/good-neighbors/)] [[source](https://github.com/jason-weirather/good-neighbors)]

### about Submodules

This primary module `pythologist` is comprised of submodules.

All of these can be cloned at once via https with the command:

```
$ git clone --recurse-submodules https://github.com/jason-weirather/pythologist.git
```

or via ssh

```
$ git clone --recurse-submodules git@github.com:jason-weirather/pythologist.git
```

Submodules will be in the `libs/` directory. For development purposes you should 

1. checkout and pull the master branch of each of these submodules
2. install each of these submodules as editable via `pip install -e .`
3. install the main `pythologist` as editable the same way `pip install -e .`

There is probably a more elegant way to use setuptools to assist in this process that I'm not doing here.



# Quickstart

To start a jupyter lab notebook with the required software as your user in your current drectory you can use the following command 

`docker run --rm -p 8888:8888 --user $(id -u):$(id -g) -v $(pwd):/work vacation/pythologist:latest`

This will start jupyter lab on port 8888 as your user and group. 

Any of the test data examples should work fine in this environment.

# Installation

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
import matplotlib.pyplot as plt

# Get the path of the test dataset
path = TestImages().raw('IrisSpatialFeatures')
# Create the storage opbject where the project will be saved
cpi = CellProjectInForm('pythologist.h5',mode='w')
# Read the project data
cpi.read_path(path,require=False,verbose=True,microns_per_pixel=0.496,sample_name_index=-1)
# Display one of the cell map images
for f in cpi.frame_iter():
    break
print(f.frame_name)
plt.imshow(f.cell_map_image(),origin='upper')
plt.show()
```

> MEL2_7
>
> ![MEL2_7_cell_map](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_7_cell_map.png?raw=true)

#### Custom region annotations from tumor and invasive margin image drawings

Another format supported for a project import is one with a custom tumor and invasive margin definition.  Similar to above, the project is organized into sample folders, and each image within each sample folder has a *tif* file defining the tumor and invasive margin.  These come in the form of a `<image name prefix>_Tumor.tif` and `<image name prefix>_Invasive_Margin.tif` for each image.  The `_Tumor.tif` is an area filled in where the tumor is, and transparent elsewhere.  The `_Invasive_Margin.tif` is a drawn line of a known width.  `steps` is used to grow the margin out that many pixels in each direction to establish an invasive margin region.  Here we also rename some markers during read-in to clean up the syntax of thresholding on binary features.


```python
from pythologist_test_images import TestImages
from pythologist_reader.formats.inform.custom import CellProjectInFormLineArea

# Get the path of the test dataset
path = TestImages().raw('IrisSpatialFeatures')
# Specify where the data read-in will be stored as an h5 object
cpi = CellProjectInFormLineArea('test.h5',mode='w')
# Read in the data (gets stored on the fly into the h5 object)
cpi.read_path(path,
              sample_name_index=-1,
              verbose=True,
              steps=76,
              project_name='IrisSpatialFeatures',
              microns_per_pixel=0.496)
for f in cpi.frame_iter():
    break
print(f.frame_name)
print('hand drawn margin')
plt.imshow(f.get_image(f.get_data('custom_images').\
    set_index('custom_label').loc['Drawn','image_id']),origin='upper')
plt.show()
print('hand drawn tumor area')
plt.imshow(f.get_image(f.get_data('custom_images').\
    set_index('custom_label').loc['Area','image_id']),origin='upper')
plt.show()
print('Mutually exclusive Margin, Tumor, and Stroma')
plt.imshow(f.get_image(f.get_data('regions').\
    set_index('region_label').loc['Margin','image_id']),origin='upper')
plt.show()
plt.imshow(f.get_image(f.get_data('regions').\
    set_index('region_label').loc['Tumor','image_id']),origin='upper')
plt.show()
plt.imshow(f.get_image(f.get_data('regions').\
    set_index('region_label').loc['Stroma','image_id']),origin='upper')
plt.show()
``` 
> MEL2_2
>
> hand drawn margin
>
> ![MEL2_2_drawn_line](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_2_drawn.png?raw=true)
>
> hand drawn tumor area
>
> ![MEL2_2_drawn_line](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_2_area.png?raw=true)
>
> Mutually exclusive Margin, Tumor, and Stroma
>
> ![MEL2_2_margin](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_2_Margin.png?raw=true)
> ![MEL2_2_tumor](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_2_Tumor.png?raw=true)
> ![MEL2_2_stroma](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_2_Stroma.png?raw=true)


### Read a project with a custom tumor mask (but no margin line)

Here we will use the mask, but not expand or subtract from it.

```python
from pythologist_test_images import TestImages
from pythologist_reader.formats.inform.custom import CellProjectInFormCustomMask
import matplotlib.pyplot as plt
path = TestImages().raw('IrisSpatialFeatures')
cpi = CellProjectInFormCustomMask('test.h5',mode='w')
cpi.read_path(path,
              microns_per_pixel=0.496,
              sample_name_index=-1,
              verbose=True,
              custom_mask_name='Tumor',
              other_mask_name='Not-Tumor')
for f in cpi.frame_iter():
    rs = f.get_data('regions').set_index('region_label')
    for r in rs.index:
        print(r)
        plt.imshow(f.get_image(rs.loc[r]['image_id']),origin='upper')
        plt.show()
    break
```
> MEL2_2
>
> Tumor
>
> ![MEL2_2_tumor](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_2_tumor-b.png?raw=true)
> 
> Not-Tumor
>
> ![MEL2_2_not_tumor](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_2_not-tumor-b.png?raw=true) 

### Quality check samples

Check general status of the CellDataFrame

```python
cdf = cpi.cdf
cdf.db = cpi
cdf.qc(verbose=True).print_results()
```

*prints the following QC metrics to stdout*

```
==========
Check microns per pixel attribute
PASS
Microns per pixel is 0.496
==========
Check storage object is set
PASS
h5 object is set
==========
Is there a 1:1 correspondence between sample_name and sample_id?
PASS
Good concordance.
Issue count: 0/2
==========
Is there a 1:1 correspondence between frame_name and frame_id?
PASS
Good concordance.
Issue count: 0/4
==========
Is there a 1:1 correspondence between project_name and project_id?
PASS
Good concordance.
Issue count: 0/1
==========
Is the same frame name present in multiple samples?
PASS
frame_name's are all in their own samples
Issue count: 0/4
==========
Are the same phenotypes listed and following rules for mutual exclusion?
PASS
phenotype_calls and phenotype_label follows expected rules
==========
Are the same phenotypes included on all images?
PASS
Consistent phenotypes
Issue count: 0/4
==========
Are the same scored names included on all images?
PASS
Consistent scored_names
Issue count: 0/4
==========
Are the same regions represented the same with an image and across images?
PASS
Consistent regions
Issue count: 0/5
==========
Are the same regions listed matching a valid region_label
PASS
regions and region_label follows expected rules
==========
Do we have any region sizes so small they should consider being excluded?
WARNING
[
    "Very small non-zero regions are included in the data['IrisSpatialFeatures', 'MEL2', 'MEL2_7', {'Margin': 495640, 'Tumor': 947369, 'Stroma': 116}]"
]
Issue count: 1/2
```

### View density plots based on cell phenotype frequencies. 

The cell phenotypes set prior to calling `cartesian` are the phenotypes available to plot.

```python
from pythologist_test_images import TestImages
from plotnine import *
proj = TestImages().project('IrisSpatialFeatures')
cdf = TestImages().celldataframe('IrisSpatialFeatures')
cdf.db = proj
cart = cdf.cartesian(verbose=True,step_pixels=50,max_distance_pixels=75)
df,cols,rngtop = cart.rgb_dataframe(red='CD8+',green='SOX10+')
shape = cdf.iloc[0]['frame_shape']
(ggplot(df,aes(x='frame_x',y='frame_y',fill='color_str'))
 + geom_point(shape='h',size=4.5,color='#777777',stroke=0.2)
 + geom_vline(xintercept=-1,color="#555555")
 + geom_vline(xintercept=shape[1],color="#555555")
 + geom_hline(yintercept=-1,color="#555555")
 + geom_hline(yintercept=shape[0],color="#555555")
 + facet_wrap('frame_name')
 + scale_fill_manual(cols,guide=False)
 + theme_bw()
 + theme(figure_size=(8,8))
 + theme(aspect_ratio=shape[0]/shape[1])
 + scale_y_reverse()
)
```

> ![Density Example](https://github.com/jason-weirather/pythologist/blob/master/images/density_plots.png?raw=true)


### View histograms of pixel intensity and the scoring of binary markers on each image

```python
from pythologist_test_images import TestImages
from plotnine import *
proj = TestImages().project('IrisSpatialFeatures')
cdf = TestImages().celldataframe('IrisSpatialFeatures')
cdf.db = proj
ch = cdf.db.qc().channel_histograms()
sub = ch.loc[(~ch['threshold_value'].isna())&(ch['channel_label']=='PDL1')]
(ggplot(sub,aes(x='bins',y='counts'))
 + geom_bar(stat='identity')
 + facet_wrap('frame_name')
 + geom_vline(aes(xintercept='threshold_value'),color='red')
 + theme_bw()
 + ggtitle('Thresholding of PDL1\ngiven image pixel intensities')
)
```

*The original component images were not available for IrisSpatialFeatures example, so pixel intensities are simulated and don't necessarily match the those which would have been used to set the original threshold values.*

> ![Histogram Example](https://github.com/jason-weirather/pythologist/blob/master/images/histogram_example.png?raw=true)

### View cell-cell contacts

```python
from pythologist_test_images import TestImages
from pythologist_reader.formats.inform.custom import CellProjectInFormCustomMask
from pythologist import SubsetLogic as SL
cpi = TestImages().project('IrisSpatialFeatures')
cdf = cpi.cdf
cdf.db = cpi
sub = cdf.loc[cdf['frame_name']=='MEL2_7'].dropna()
cont = sub.contacts().threshold('CD8+','CD8+/contact').contacts().threshold('SOX10+','SOX10+/contact')
cont = cont.threshold('CD8+','SOX10+/contact',
                      positive_label='CD8+ contact',
                      negative_label='CD8+').\
    threshold('SOX10+','CD8+/contact',
              positive_label='SOX10+ contact',
              negative_label='SOX10+')
schema = [
    {'subset_logic':SL(phenotypes=['OTHER']),
     'edge_color':(50,50,50,255),
     'watershed_steps':0,
     'fill_color':(0,0,0,255)
    },
    {'subset_logic':SL(phenotypes=['SOX10+']),
     'edge_color':(166,206,227,255),
     'watershed_steps':0,
     'fill_color':(0,0,0,0)
    },
    {'subset_logic':SL(phenotypes=['CD8+']),
     'edge_color':(253,191,111,255),
     'watershed_steps':0,
     'fill_color':(0,0,0,0)
    },
    {'subset_logic':SL(phenotypes=['CD8+ contact']),
     'edge_color':(253,191,111,255),
     'watershed_steps':0,
     'fill_color':(255,127,0,255)
    },
    {'subset_logic':SL(phenotypes=['SOX10+ contact']),
     'edge_color':(166,206,227,255),
     'watershed_steps':0,
     'fill_color':(31,120,180,255)
    }
]
sio = cont.segmentation_images().build_segmentation_image(schema,background=(0,0,0,255))
sio.write_to_path('test_edges',overwrite=True)
```

> MEL2_7
>
> ![Visualize Contacts](https://github.com/jason-weirather/pythologist/blob/master/images/MEL2_7-contacts.png?raw=true)

*Image is zoomed-in and cropped to show the contours better.*

### Merge CellDataFrames that have the same image segmentations but different scored calls

This happens frequently because current InForm exports only permit two features to be scored per export

```python
merged,fail = cdf1.merge_scores(cdf2,on=['sample_name','frame_name','x','y'])
```


### Show names of the binary 'scored_calls'
```python
cdf.scored_names
```

> ['PD1', 'PDL1']

### Show phenotypes
```python
cdf.phenotypes
```

> ['CD8+', 'OTHER', 'SOX10+']

### Show regions
```python
cdf.regions
```

> ['Margin', 'Stroma', 'Tumor']

### Combine two or more phenotypes into one or rename a phenotype
```python
collapsed = cdf.collapse_phenotypes(['CD8+','OTHER'],'non-Tumor')
collapsed.phenotypes
```

> ['SOX10+', 'non-Tumor']

### Rename a region

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

### Create an image of cell-cell contacts between features of interest

```python

```

# Check outputs against IrisSpatialFeatures outputs

To ensure we are generating expected outs we can check against the outputs of IrisSpatialFeatures [[github](https://github.com/gusef/IrisSpatialFeatures)]. 

* Jupyter Notebook: [Test against IrisSpatialFeatures outputs](https://github.com/jason-weirather/pythologist/blob/master/notebooks/Test%20against%20IrisSpatialFeatures%20outputs.ipynb)
