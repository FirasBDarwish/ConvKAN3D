import torch
import torch.nn.functional as F
import warnings
import unfoldNd
from enum import Enum

from ConvKAN3D.efficient_kan import KANLinear

"""
The parameters stride, padding, dilation can either be:

    + a single int - in which case the same value is used for the depth, height and width dimension
    + a tuple of three ints - in which case, the first int is used for the depth dimension, the second int for the height dimension and the third int for the width dimension
"""
class effConvKAN3D(torch.nn.Module):
    """
    This is an implementation of convolutional layers using Kolmogorov-Arnold Networks (KANs)
    """
    def __init__(self,
                in_channels, # number of input channels
                out_channels, # number of output channels
                kernel_size, # the size of the kernel, MUST be a 3-tuple of type (depth, height, width) kernel_sizes
                stride=1, # controls stride for cross-correlation
                padding=0, # controls the amount of padding applied to the input
                dilation=1, #  controls the spacing between the kernel points; also known as the à trous algorithm
                padding_mode='zeros', # 'zeros' ONLY

                # refer to https://github.com/Blealtan/efficient-kan for what these variables expect
                grid_size=5,
                spline_order=3,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                enable_standalone_scale_spline=True,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1], device=None, dtype=None):
        
        if not effConvKAN3D._is_3_tuple(kernel_size):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        super(effConvKAN3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode
        if(self.padding_mode != 'zeros'):
            warnings.warn("Warning! Padding_mode is assumed to be 'zeros'. Instable results may arise")

        self.unfold = unfoldNd.UnfoldNd(self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        
        self.linear = KANLinear(self.in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2],
                                self.out_channels,
                                grid_size,
                                spline_order,
                                scale_noise,
                                scale_base,
                                scale_spline,
                                enable_standalone_scale_spline,
                                base_activation,
                                grid_eps,
                                grid_range)

    def forward(self, x):
        """
        forward propagation,
        input assumed to be of form: (N,Cin,Depth,Height,Width)
        """

        assert x.dim() == 5
        batch_size, in_channels, depth, height, width = x.size()
        assert in_channels == self.in_channels

        # Unfold the input tensor to extract flattened sliding blocks from a batched input tensor.
        # Input:  [batch_ size, in_channels, depth, height, width]
        # Output: [batch_size, in_channels * ∏(kernel_size), L], where L is the total number of such blocks.
        blocks = self.unfold(x)

        # Move the flattened sliding blocks into the last dimension such that:
        # Input: [batch_size, in_channels * ∏(kernel_size), L]
        # Output: [batch_size, L, in_channels *  ∏(kernel_size)]
        blocks = blocks.transpose(1,2) # left to right dimensions: 0, 1, 2 (we are swapping 1 and 2)

        # reshape such that batch_size and number of blocks (L) are mixed:
        # Input: [batch_size, L, in_channels *  ∏(kernel_size)]
        # Output: [batch_size * L, in_channels *  ∏(kernel_size)]
        blocks = blocks.reshape(-1, self.in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2])

        # Apply KAN Linear layer on all blocks (across all batches)
        # Input: [batch_size * L, in_channels *  ∏(kernel_size)]
        # Output: [batch_size * L, out_channels]
        out = self.linear(blocks)

        # Reshape the output to retrieve batch dimension
        # Input: [batch_size * L, out_channels]
        # Output: [batch_size, L, out_channels]
        out = out.reshape(batch_size, -1, out.shape[-1])

        # Transpose to move the number of blocks to the last dimension
        # Input: [batch_size, L, out_channels]
        # Output: [batch_size, out_channels, L]
        out = out.transpose(1,2)

        # ensure all other parameters for convolution are tuples to find dimensionality of output feature map
        if(not effConvKAN3D._is_3_tuple(self.stride)):
            self.stride = (self.stride, self.stride, self.stride)
        if(not effConvKAN3D._is_3_tuple(self.padding)):
            self.padding = (self.padding, self.padding, self.padding)
        if(not effConvKAN3D._is_3_tuple(self.dilation)):
            self.dilation = (self.dilation, self.dilation, self.dilation)
        
        depth_out = ( (depth + 2*self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] ) + 1
        height_out = ( (height + 2*self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] ) + 1
        width_out = ( (width + 2*self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) // self.stride[2] ) + 1

        # Reshape to retrieve depth, height, and width of output feature map
        # Input: [batch_size, out_channels, L]
        # Output: [batch_size, out_channels, depth_out, height_out, width_out]
        out = out.view(batch_size, self.out_channels, depth_out, height_out, width_out)

        return out

    def _is_3_tuple(var):
        """
        Helper function to check if a variable is a 3-tuple
        """
        return isinstance(var, tuple) and len(var) == 3
