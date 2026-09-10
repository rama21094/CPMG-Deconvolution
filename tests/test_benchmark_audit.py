import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
from cpmg_ml.benchmark_audit import audit


class AuditTests(unittest.TestCase):
    def test_exact_cross_split_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split,values in [('train',[1,2,2]),('val',[1,3]),('test',[2,4])]:
                with h5py.File(root/f'{split}_0000.h5','w') as h:
                    for key in ('metadata','nu_cp','r2_with_j','r2_no_j'):
                        h[key]=np.array(values,dtype=np.float32)[:,None]
                    h.attrs['t_relax']=.04
            result=audit(root)
            self.assertEqual(result['excluded_duplicate_row_indices'],{'train':[2],'val':[0],'test':[0]})
            self.assertEqual(result['duplicate_occurrences_by_split'],{'train->train':1,'train->val':1,'train->test':1})


if __name__=='__main__':
    unittest.main()
