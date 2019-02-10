import json
class SubsetLogic(dict):
    def __init__(self,*args,**kwcopy):
        if 'label' not in kwcopy: kwcopy['label'] = None
        if 'phenotypes' not in kwcopy: kwcopy['phenotypes'] = []
        if 'scored_calls' not in kwcopy: kwcopy['scored_calls'] = {}
        super(SubsetLogic,self).__init__(*args,**kwcopy)
        self.itemlist = super(SubsetLogic,self).keys()
        return

    def to_json(self):
        return json.dumps(self)
    
    @property
    def label(self):
        return self['label']
    @label.setter
    def label(self,value):
        self['label'] = value
    @property
    def phenotypes(self):
        return self['phenotypes'].copy()    
    @phenotypes.setter
    def phenotypes(self,value):
        self['phenotypes'] = value
    @property
    def scored_calls(self):
        return self['scored_calls'].copy()    
    @scored_calls.setter
    def scored_calls(self,value):
        self['scored_calls'] = value

