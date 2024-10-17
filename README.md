Zamba 2, HQQ-ified.
===================

Notebook shows that it's possible to quantize 7b model of zamba 2
https://huggingface.co/Zyphra/Zamba2-7B-Instruct

# Model code fixes

The model code is replica of

https://github.com/Zyphra/transformers_zamba2/tree/main/src/transformers/models/zamba2

with minimum modifications which made it work.

One noticeable is `_get_states_from_cache` in mamba2_layer: it was changed from

    ssm_state = torch.zeros(
        batch_size,
        self.nheads,
        self.headdim,
        self.d_state,
        device=self.in_proj[0].weight.device,
        dtype=torch.bfloat16
    )

to 

    ssm_state = torch.zeros(
        batch_size,
        self.nheads,
        self.headdim,
        self.d_state,
        device=self.conv1d.weight.device,
        dtype=torch.bfloat16
    )

as originally I used `HQQLinear` instead of `HQQLinearTorchWeightOnlynt4`
and `HQQLinear` lacks `weight`.

Others changes are my vim having opinion about autoformatting and changing import from .mamba_layer2

Another noticeable difference is on inference side:

`HybridMambaAttentionDynamicCache` doesn't inherit from `transformers.Cache`, which breaks transformers `4.45.2` if I try to use it with `generate` function.

I haven't check long-time use of cache(i.e. generating response, then using cache to call generate again)


# HQQ utilization

HQQ doesn't requires finetuning, which makes it a perfect tool for making quants of the model with novel architecture. 

One most important thing about HQQ is that on my laptop using 
`torchao_int4` is mandatory: ATEN, PYTORCH and PYTORCH_COMPILE are slow even at 1.2B model. AO_INT4 works about as quickly as does raw BF16.

# Doing quants at 16 GB vrams.

Making quants out of `linear_fc1`, `linear_fc2` in MLP blocks requires lots of memory to the point I'm getting OoM. However if mamba layers are moved from GPU to CPU during MLP quantification, then moved back, 16GB of vram is enough.
