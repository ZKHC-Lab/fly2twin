"""Backend dispatch and precision contract; no compiled FA dependency for CPU tests."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vggt_omega.models.layers import flash_attention
from vggt_omega.models.layers.attention import sdpa


class FakeExtension:
    def fwd(self,q,k,v,out,cuq,cuk,nq,nk,dropout,scale,zero,causal,softmax,splits,generator):
        batch=cuq.numel()-1
        heads,dim=q.shape[1:]
        assert dropout==0 and not causal and not softmax
        qp=q.reshape(batch,nq,heads,dim).transpose(1,2)
        kp=k.reshape(batch,nk,heads,dim).transpose(1,2)
        vp=v.reshape(batch,nk,heads,dim).transpose(1,2)
        out.copy_(F.scaled_dot_product_attention(qp,kp,vp).transpose(1,2).reshape_as(out))
        return ()


class AttentionBackendTest(unittest.TestCase):
    @torch.inference_mode()
    def test_fused_bridge_preserves_layout_and_dtype(self):
        for dtype in (torch.float16,torch.bfloat16):
            q=torch.randn(2,3,13,64).to(dtype)
            k=torch.randn(2,3,19,64).to(dtype)
            v=torch.randn_like(k)
            with patch.object(flash_attention,'extension',return_value=FakeExtension()):
                got=sdpa(q,k,v,backend='flash_attn_v1')
            torch.testing.assert_close(got,F.scaled_dot_product_attention(q,k,v),rtol=0,atol=0)

    def test_fp32_heads_keep_native_attention(self):
        q=torch.randn(1,2,13,64)
        with patch.object(flash_attention,'extension',side_effect=AssertionError('FP32 must not load FA')):
            got=sdpa(q,q,q,backend='flash_attn_v1')
        torch.testing.assert_close(got,F.scaled_dot_product_attention(q,q,q),rtol=0,atol=0)

    def test_unknown_backend_and_unsupported_training_fail(self):
        q=torch.randn(1,2,13,64,dtype=torch.bfloat16,requires_grad=True)
        with self.assertRaisesRegex(ValueError,'unknown attention'):
            sdpa(q,q,q,backend='typo')
        with self.assertRaisesRegex(RuntimeError,'inference-only'):
            sdpa(q,q,q,backend='flash_attn_v1')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA autocast test')
    @torch.inference_mode()
    def test_fp32_inside_bf16_autocast_is_not_an_fp32_head(self):
        q=torch.randn(2,4,31,64,device='cuda',dtype=torch.float32)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            with patch.object(flash_attention,'extension',return_value=FakeExtension()):
                got=sdpa(q,q,q,backend='flash_attn_v1')
            ref=F.scaled_dot_product_attention(q,q,q)
        self.assertEqual(got.dtype,torch.bfloat16)
        torch.testing.assert_close(got,ref,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
