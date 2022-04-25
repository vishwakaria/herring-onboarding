from torch.nn.modules import Module
from torch.autograd import Variable
import torch
import torch.distributed as dist


def vocal_errors(func):
    """Errors emitted from hooks are swallowed by PyTorch
    and show up as unrecognizable RuntimeError in autograd.
    Lets print them first."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(
                    "Error in grad hook."
            )
            print("Error: ", e, type(e))
            import traceback
            traceback.print_stack()
            raise e

    return wrapper

class DistributedDataParallel(Module):
    def __init__(self,
                 module,
                 device_ids=None,
                 output_device=None,
                 dim=0,
                 broadcast_buffers=True,
                 process_group=None,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False,
                 gradient_as_bucket_view=False,
                 gradient_accumulation_steps=None):

        super(DistributedDataParallel, self).__init__()

        self.module = module
        self._trainable_params = None
        self._num_gpus = dist.get_world_size()
        self._rank = dist.get_rank()


        #Register hook to receive gradients
        for param in self.module.parameters():
            param.register_hook(lambda grad: self._grad_multiply_hook(grad))
        
        #Broadcast parameters to make sure all devices have the same params to begin with
        for _, param in enumerate(self.module.parameters()):
            print('param on rank {} before broadcast: {}'.format(self._rank, param.detach().cpu().numpy()))
            self._broadcast(param)
            print('param on rank {} after broadcast: {}'.format(self._rank, param.detach().cpu().numpy()))
    
   
    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        self._final_callback_required = False
        return result

    def _get_trainable_params(self):
        if self._trainable_params is None:
            self._trainable_params = [
                param for param in self.module.parameters()
                if param.requires_grad
            ]
        return self._trainable_params

    @vocal_errors
    def _grad_multiply_hook(self, grad):
        grad = 0.1 * grad
        return grad

    def _broadcast(self, grad, rootRank=0):
        if self._rank == 0:
            dist.broadcast(grad, rootRank)
