import torch

def get_real_mask(emask, edge_constraints, num_edges):
    '''
    Given the masking value of an edge, computes the actual masking value
    by multiplying (1-mask) of each edge that would cross it.
    '''
    real_mask = torch.ones(num_edges)
    for i in range(num_edges):
        real_mask[i] = emask[i]
        if i in edge_constraints.keys():
            for k in edge_constraints[i]:
                real_mask[i] *= (1-emask[k])
    return real_mask

def scalar_to_tensor(value, size):
    '''
    Turns 'value' into a torch.FloatTensor with 'size' identical entries.
    Simply returns 'value' if it is already a torch.FloatTensor.
    '''
    try:
        len(value)
    except:
        value = torch.Tensor([value]*size)
    assert(len(value)==size)

    return value

class SuperSpike(torch.autograd.Function):
    '''
    Surrogate function from

    Implementation taken from Norse (https://github.com/norse/norse).
    '''
    @staticmethod
    @torch.jit.ignore
    def forward(ctx, input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return (input_tensor > 0.)*1.

    @staticmethod
    @torch.jit.ignore
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad = grad_input / (alpha * torch.abs(inp) + 1.0).pow(
            2
        ) 
        return grad, None

@torch.jit.ignore
def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    return SuperSpike.apply(x, alpha)

@torch.jit.script
def heaviside(data):
    '''
    A heaviside step function that truncates numbers <= 0 to 0 and everything else to 1.

    Implementation taken from Norse (https://github.com/norse/norse).
    '''
    return torch.gt(data, torch.as_tensor(0.0)).to(data.dtype)  # pragma: no cover
