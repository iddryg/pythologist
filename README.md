# pythologist

Read files from inForm software by PerkinElmer.

## Common tasks

### Reading in data

```python
import pythologist

fname = 'data directory'
raw = pythologist.read_inForm(fname)
```

Directory names are infered from the last folder name.  If this is not actually the sample name, you can safely use the folder path as a sample name, or specify a `sample_index` in argument.

*Use the second to last folder as the sample name*

```python
import pythologist

fname = 'data directory'
raw = pythologist.read_inForm(fname,sample_index=1)
```

*Use the folder path as the sample name*

```python
import pythologist

fname = 'data directory'
raw = pythologist.read_inForm(fname)
# extract a copy of the data frame
temp = raw.df
# fix the name
temp['sample'] = temp['folder']
# recreate an InFormCellFrame and set your microns per pixel attribute
raw = pythologist.InFormCellFrame(temp,mpp=raw.mpp)
```

### Quality check samples

```python
import pythologist
from pythologist.test import InFormCellFrameTest

raw = pythologist.read_inForm('myfolder')
test = InFormCellFrameTest(raw)
test.check_overlapping_samples()
```

```python
test.check_overlapping_frames()
```

```python
test.check_scored_stain_consistency()
```

```python
test.check_phenotype_consistency()
```

```python
test.check_tissue_consistency()
```

```python
test.scored_stain_compartment()
```


### Show the stains present
```python
raw.all_stains
```

### Rename a tisssue

Rename *TUMOR* to *Tumor*

```python
raw = raw.rename_tissue('TUMOR','Tumor')
```

### Threshold a phenotype

Make *CYTOK* into *CYTOK PDL1+* and *CYTOK PDL1-*

```python
raw_thresh = raw.threshold('PDL1 (Opal 520)','CYTOK','PDL1')
```

### Double threshold

```python
CD68_CD163 = raw.threshold('CD163 (Opal 690)','CD68','CD163')
CD68_CD163pos_PDL1 = CD68_CD163.threshold('PDL1 (Opal 520)','CD68 CD163+','PDL1')
```

### Get per frame counts

```python
frame_counts = raw.frame_counts
frame_counts
```

write those counts out to a csv file

```python
frame_counts.to_csv('my_frame_counts.csv')
```

### Get per sample counts

```python
sample_counts = raw.sample_counts
sample_counts
```

write those counts out to a csv file

```python
sample_counts.to_csv('my_sample_counts.csv')
```

### Save gated-relabeled data as an inFrom compatible project

1. make a copy of your data

either copy the folder in your file explorer or from command line i.e.

`$ cp -r myfolder myfolder2`

2. write your new data into the file

```python
CD68_CD163pos.write_inForm('myfolder2',overwrite=True)
```

Now the project myfolder2 can be used in IrisSpatialFeatures