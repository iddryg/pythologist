# pythologist

*Read and analyze cell image data.*

Pythologist provides tools for 1) reading exports from InForm software into a common storage format, and 2) analyzing image data where cells have been segemented and annotated.

## Docker quickstart

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

## Common tasks

### Reading in a project composed of InForm exports

```python
from pythologist_reader.formats.inform.sets import CellProjectInForm
oname = 'my_inform_export_directory'
cpi = CellProjectInForm(oname,'w')
cpi.read_path(fname,project_name='MyProject',name_index=-1,verbose='True',microns_per_pixel=0.496)
cdf = cpi.cdf
```

The assumption here is that the exports are grouped so that sample folders contain one or more image exports, and that sample name can be inferred from the last folder name.

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

generate counts and fractions of the current phenotypes and export them to a csv

```python
cdf.counts().frame_counts().to_csv('my_frame_counts.csv')
```

### Get per sample counts

```python
cdf.counts().sample_counts().to_csv('my_sample_counts.csv')
```

