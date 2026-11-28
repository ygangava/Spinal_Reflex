"""Torch version of the LIF step, for training only (Phase D).

Must compute EXACTLY what snn.lif_step computes in the forward pass. The only new thing
is the backward pass: the spike's gradient is a surrogate (a smooth bump), not zero.
"""
import torch


class SpikeFn(torch.autograd.Function):
    """Heaviside spike with a fast-sigmoid surrogate gradient.

    Input x = v - threshold. -- membrane's distance from threshold
    forward:  spike = 1 if x >= 0 else 0                  (exactly like lif_step)
    backward: d spike / dx ≈ 1 / (1 + SLOPE·|x|)²         (pretend the step was a smooth bump)
    """
    SLOPE = 10.0

    @staticmethod
    def forward(ctx, x):
        # save x for backward (ctx.save_for_backward)
        # ctx is pytorch storage 
        ctx.save_for_backward(x)
        r = (x>=0).to(x.dtype)  # makes it a 0.0 or 1.0
        return r

    @staticmethod
    def backward(ctx, grad_out):
        # get x back (ctx.saved_tensors)
        x, = ctx.saved_tensors
        # return grad_out × surrogate(x)
        #       (this is the MLP's  dpre = da · f'(pre)  line, with f' = the bump)
        # slope dependent on on distnace from threshold on each side
        surr = 1/(1+SpikeFn.SLOPE * x.abs()) ** 2
        # gradient arriving from later calculations × this operation’s slope
        chain_rule_out = grad_out * surr
        return chain_rule_out


def lif_step_torch(v, current, beta, threshold=1.0):
    """Same 3 steps as snn.lif_step. v, current: (batch, 64) tensors. Returns (v_new, spike)."""
    # leak + integrate
    v_new = beta * v + current
    spike = SpikeFn.apply(v_new - threshold)
    # reset -- gradient does not flow through the reset term, uses pike value but detatch grad
    v_new = v_new - threshold * spike.detach()
    
    
    return v_new, spike
