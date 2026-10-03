import tempfile
from pathlib import Path
import unittest
from research.programme_closure import resume_confirmation as R


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.identity=dict(plan='fixed',checkpoint='fixed',runner='fixed',seed=8101)
        self.origins=[dict(market='A',date=str(i),target=float(i)) for i in range(5)]

    def block(self,start,end,identity=None):
        rows=[dict(x,h=1.,raw_quantiles=list(range(-4,5))) for x in self.origins[start:end]]
        R.E.save(self.path/('block_%06d.json'%start),dict(identity=identity or self.identity,start=start,rows=rows))
        return rows

    def test_resume_preserves_order_and_values(self):
        expected=self.block(0,2)+self.block(2,5)
        self.assertEqual(R.load_blocks(self.path,self.origins,self.identity),expected)

    def test_partial_atomic_temporary_file_is_not_a_saved_forecast(self):
        expected=self.block(0,2)
        (self.path/'block_000002.json.tmp').write_text('{')
        self.assertEqual(R.load_blocks(self.path,self.origins,self.identity),expected)

    def test_wrong_model_or_plan_rejected(self):
        self.block(0,2,dict(self.identity,checkpoint='wrong'))
        with self.assertRaisesRegex(ValueError,'identity'):R.load_blocks(self.path,self.origins,self.identity)

    def test_missing_block_rejected(self):
        self.block(0,2);self.block(3,5)
        with self.assertRaisesRegex(ValueError,'sequence'):R.load_blocks(self.path,self.origins,self.identity)

    def test_changed_target_rejected(self):
        self.block(0,2);self.origins[1]['target']=999.
        with self.assertRaisesRegex(ValueError,'origin'):R.load_blocks(self.path,self.origins,self.identity)


if __name__=='__main__':unittest.main()
