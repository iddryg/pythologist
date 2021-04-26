# Test all the inform read functions
import unittest, os, shutil
from tempfile import mkdtemp, gettempdir

from pythologist_reader.formats.inform.frame import CellFrameInForm
from pythologist_reader.formats.inform.sets import CellSampleInForm, CellProjectInForm

class InFormFrameTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Get the temporary directors and run the command"""
        from pythologist_test_images import TestImages
        e1 = os.path.join(TestImages().raw('Tiny'),'E1')
        cfi = CellFrameInForm()
        print('==========')
        print("reading in InForm single frame")
        print('----------')
        cfi.read_raw(frame_name='E1_1',
                     cell_seg_data_file=os.path.join(e1,'E1_1_cell_seg_data.txt'),
                 score_data_file=os.path.join(e1,'E1_1_score_data.txt'),
                 tissue_seg_data_file=None,
                 binary_seg_image_file=os.path.join(e1,'E1_1_binary_seg_maps.tif'),
                 component_image_file=None,
                 verbose=True,
                 channel_abbreviations=None,
                 require=False)
        self.frame = cfi
        return
    # Test the InForm at the Frame level
    def test_1(self):
        print(type(self.frame))
        self.assertEqual(1,1)
class InFormSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Get the temporary directors and run the command"""
        from pythologist_test_images import TestImages
        e1 = os.path.join(TestImages().raw('Tiny'),'E1')
        print('==========')
        print("reading in InForm single sample")
        print('----------')
        csi = CellSampleInForm()
        csi.read_path(e1,sample_name='E1',
                 verbose=True,
                 channel_abbreviations=None,
                 require=False)
        self.sample = csi
        return
    # Test the InForm at the Frame level
    def test_1(self):
        print(type(self.sample))
        self.assertEqual(1,1)
class InFormProjectTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Get the temporary directors and run the command"""
        from pythologist_test_images import TestImages
        p1 = TestImages().raw('Tiny')
        print('==========')
        print("reading in InForm full project")
        print('----------')
        self.dirpath = mkdtemp(prefix="pythologist_reader.",dir=gettempdir())
        self.h5file = os.path.join(self.dirpath,'pythologist.h5')
        #self.outdir = os.path.join(self.dirpath,'output')
        cpi = CellProjectInForm(self.h5file,mode='w')
        cpi.read_path(p1,project_name='inform_simulated',
                 verbose=True,
                 channel_abbreviations=None,
                 sample_name_index=-1,
                 require=False)
        self.project = cpi
        return
    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.dirpath)
        return
    # Test the InForm at the Frame level
    def test_1(self):
        print(type(self.project))
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()

