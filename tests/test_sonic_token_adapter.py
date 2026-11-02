import unittest
import torch
from DiT4DiT.evaluation.sonic_token_adapter import restore_internal_tokens,decode_feature_dict


class TokenAdapterTests(unittest.TestCase):
    def test_index_order_and_noncontiguous(self):
        flat=torch.arange(2*3*128,dtype=torch.float32).reshape(2,3,128)[...,::2]
        self.assertFalse(flat.is_contiguous())
        tokens=restore_internal_tokens(flat,num_tokens=2,token_dim=32)
        self.assertEqual(tokens.shape,(2,3,2,32))
        for b in range(2):
            for h in range(3):
                for n in range(2):
                    for d in range(32):
                        self.assertEqual(tokens[b,h,n,d],flat[b,h,n*32+d])
        torch.testing.assert_close(tokens.flatten(-2),flat,atol=0,rtol=0)

    def test_invalid_contract(self):
        for flat in (torch.zeros(2,64),torch.zeros(2,3,63),torch.full((2,3,64),float('nan')),torch.zeros(2,3,64,dtype=torch.int64)):
            with self.assertRaises(ValueError):restore_internal_tokens(flat,num_tokens=2,token_dim=32)

    def test_feature_order_is_declared_order(self):
        inp=torch.arange(12,dtype=torch.float32).reshape(1,2,6)
        a=decode_feature_dict(lambda x:x,{'token':inp},['token'],{'heading':2,'command':4})
        b=decode_feature_dict(lambda x:x,{'token':inp},['token'],{'command':4,'heading':2})
        torch.testing.assert_close(a['command'],inp[...,2:])
        torch.testing.assert_close(b['command'],inp[...,:4])
        with self.assertRaises(ValueError):decode_feature_dict(lambda x:x,{'token':inp},['token'],{'command':5})
        with self.assertRaises(KeyError):decode_feature_dict(lambda x:x,{},['token'],{'command':6})

    def test_input_feature_order(self):
        a=torch.ones(1,2,3);b=2*a
        fields=decode_feature_dict(lambda x:x,{'a':a,'b':b},['b','a'],{'first':3,'second':3})
        torch.testing.assert_close(fields['first'],b)
        torch.testing.assert_close(fields['second'],a)

if __name__=='__main__':unittest.main()
