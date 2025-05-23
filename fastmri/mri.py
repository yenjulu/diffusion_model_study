"""
This module implements MRI operators

Author: Zhengguo Tan <zhengguo.tan@gmail.com>
"""

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import torch
import torch.nn as nn
from typing import Optional, Tuple


def fftc(input: torch.Tensor | np.ndarray,
         axes: Optional[Tuple] = (-2, -1),
         norm: Optional[str] = 'ortho'):

        if isinstance(input, np.ndarray):
            tmp = np.fft.ifftshift(input, axes=axes)
            tmp = np.fft.fftn(tmp, axes=axes, norm=norm)
            output = np.fft.fftshift(tmp, axes=axes)

        elif isinstance(input, torch.Tensor):
            tmp = torch.fft.ifftshift(input, dim=axes)
            tmp = torch.fft.fftn(tmp, dim=axes, norm=norm)
            output = torch.fft.fftshift(tmp, dim=axes)

        return output

def ifftc(input: torch.Tensor | np.ndarray,
          axes: Optional[Tuple] = (-2, -1),
          norm: Optional[str] = 'ortho'):

        if isinstance(input, np.ndarray):
            tmp = np.fft.fftshift(input, axes=axes)
            tmp = np.fft.ifftn(tmp, axes=axes, norm=norm)
            output = np.fft.ifftshift(tmp, axes=axes)

        elif isinstance(input, torch.Tensor):
            tmp = torch.fft.fftshift(input, dim=axes)
            tmp = torch.fft.ifftn(tmp, dim=axes, norm=norm)
            output = torch.fft.ifftshift(tmp, dim=axes)

        return output

def sense1(input_kspace, sens_maps, axes=(-2, -1)):
    """
    Parameters
    ----------
    input_kspace : ncoil x nrow x ncol
    sens_maps :  ncoil x nrow x ncol 
    axes : The default is (-2,-1).

    Returns
    -------
    sense1 image

    """
    image_space = ifftc(input_kspace, axes=axes, norm='ortho') 
    sense1_image = np.sum(image_space * np.conj(sens_maps), axis=0)  #  nrow x ncol 

    return sense1_image


class SenseOp_NUFFT():
    """
    Sensitivity Encoding (SENSE) Operators

    Reference:
        * Pruessmann KP, Weiger M, Börnert P, Boesiger P.
          Advances in sensitivity encoding with arbitrary k-space trajectories.
          Magn Reson Med (2001).
    """
    def __init__(self,
                 coil: torch.Tensor | np.ndarray,
                 mask: torch.Tensor | np.ndarray,
                 dcf : Optional[bool] = False, # Density Compensation Function
                 verbose=False,
                 normalization=False,
                 device=torch.device('cpu')):
        """
        Args:
            coil: [1, N_coil, N_y, N_x]
            mask: [N_frames, N_spokes, N_samples, 2]  # radial trajectory
        """

        if isinstance(coil, np.ndarray):
            coil = torch.from_numpy(coil)

        coil = coil.to(device)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        mask = mask.to(device)

        self.coil = coil
        self.mask = mask
        self.dcf = dcf
        self.verbose = verbose
        self.normalization = normalization
        self.device = device

        N_batch = coil.shape[0]
        assert 1 == N_batch

        assert 2 == mask.shape[-1]

        self.N_samples = mask.shape[-2]
        self.N_spokes  = mask.shape[-3]
        self.N_frames  = mask.shape[-4]

        base_res = int(self.N_samples//2)

        im_size = [base_res] * 2
        grid_size = [self.N_samples] * 2

        self.ishape = [self.N_frames] + im_size  # [N_frames, base_res, base_res]

        self.im_size = im_size
        self.NUFFT_FWD = tkbn.KbNufft(im_size=im_size, grid_size=grid_size)
        self.NUFFT_ADJ = tkbn.KbNufftAdjoint(im_size=im_size, grid_size=grid_size)

        self.NUFFT_FWD = self.NUFFT_FWD.to(coil.device)
        self.NUFFT_ADJ = self.NUFFT_ADJ.to(coil.device)

    def _get_normalization_scale(self, nrm_0, nrm_1, output_dim):

        if self.normalization:
            if torch.all(nrm_1 == 0):
                nrm_1 = nrm_1 + 1E-6
            scale = nrm_0 / nrm_1
            for _ in range(output_dim-1):
                scale = scale.unsqueeze(1)
        else:
            scale = 1

        return scale

    def fwd(self, input) -> torch.Tensor:
        """
        SENSS Forward Operator: from image to k-space
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        input = input.to(self.device)

        if torch.is_floating_point(input):
            input = input + 1j * torch.zeros_like(input)

        N_batch, N_coil, N_y, N_x = self.coil.shape

        nrm_0 = torch.linalg.norm(input, dim=(-2, -1)).flatten()

        output = []

        for t in range(self.N_frames):

            traj_t = torch.reshape(self.mask[..., t, :, :, :], (-1, 2)).transpose(1, 0)
            imag_t = torch.squeeze(input[..., t, :, :]).unsqueeze(0).unsqueeze(0)

            grid_t = self.NUFFT_FWD(imag_t, traj_t, smaps=self.coil)

            if self.verbose:
                print('> frame ', str(t).zfill(2))
                print('  traj shape: ', traj_t.shape)
                print('  imag shape: ', imag_t.shape)
                print('  grid shape: ', grid_t.shape)

            output.append(grid_t.detach().cpu().numpy())

        output = torch.tensor(np.array(output)).to(self.coil)
        nrm_1 = torch.linalg.norm(output, dim=(-2, -1)).flatten()

        scale = self._get_normalization_scale(nrm_0, nrm_1, output.dim())
        output = output * scale

        return output

    def adj(self, input) -> torch.Tensor:
        """
        SENSE Adjoint Operator: from k-space to image
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        input = input.to(self.device)

        if torch.is_floating_point(input):
            input = input + 1j * torch.zeros_like(input)

        nrm_0 = torch.linalg.norm(input, dim=(-2, -1)).flatten()

        output = []

        for t in range(self.N_frames):

            traj_t = torch.reshape(self.mask[..., t, :, :, :], (-1, 2)).transpose(1, 0)

            grid_t = input[t]

            # density compensation function
            if self.dcf:
                comp_t = (traj_t[0, ...]**2 + traj_t[1, ...]**2)**0.5 + 1E-5
                comp_t = comp_t.reshape(1, -1)
                # comp_t = tkbn.calc_density_compensation_function(ktraj=traj_t, im_size=self.im_size)
            else:
                comp_t = 1.

            imag_t = self.NUFFT_ADJ(grid_t * comp_t, traj_t, smaps=self.coil)

            if self.verbose:
                print('> frame ', str(t).zfill(2))
                print('  traj shape: ', traj_t.shape)
                print('  grid shape: ', grid_t.shape)
                print('  grid shape: ', imag_t.shape)

            output.append(imag_t.detach().cpu().numpy().squeeze())

        output = torch.tensor(np.array(output)).to(self.coil)
        nrm_1 = torch.linalg.norm(output, dim=(-2, -1)).flatten()

        scale = self._get_normalization_scale(nrm_0, nrm_1, output.dim())
        output = output * scale

        return output

class SenseOp():
    """
    Sensitivity Encoding (SENSE) Operators

    Reference:
        * Pruessmann KP, Weiger M, Börnert P, Boesiger P.
          Advances in sensitivity encoding with arbitrary k-space trajectories.
          Magn Reson Med (2001).
    """
    def __init__(self,
                 coil: torch.Tensor | np.ndarray,
                 mask: torch.Tensor | np.ndarray,
                 traj: Optional[torch.Tensor | np.ndarray] = None):
        """
        Args:
            coil: [N_batch, N_coil, N_y, N_x]
            mask: [N_batch, N_y, N_x]
            traj:
        """

        if isinstance(coil, np.ndarray):
            coil = torch.from_numpy(coil)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        if (traj is not None) and isinstance(traj, np.ndarray):
            traj = torch.from_numpy(traj)


        self.coil = coil
        self.mask = mask
        self.traj = traj

    def fwd(self, input) -> torch.Tensor:
        """
        SENSE Forward Operator: from image to k-space
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        N_batch, N_coil, N_y, N_x = self.coil.shape

        coils = torch.swapaxes(self.coil, 0, 1)
        coils = coils * input
        # coil =  N_coil, N_batch, N_y, N_x
        # input = N_batch, N_y, N_x --> [1, N_batch, N_y, N_x] complex64
        
        kfull = fftc(coils, norm='ortho')  # coil here is coil-specific input img 

        if self.traj is None:
            # Cartesian sampling
            output = torch.swapaxes(self.mask * kfull, 0, 1)
            # output is masked coil-specific input img
            
        else:
            # Radial sampling
            output = self.radial_sample(kfull)

        return output

    def radial_sample(self, kspace):
        """ Sample k-space data along radial trajectories. """
        N_batch, N_coil, N_y, N_x = kspace.shape
        center_y, center_x = N_y // 2, N_x // 2

        # Assuming self.traj contains angles in radians for each spoke
        num_spokes = self.traj.shape[0]
        output = torch.zeros_like(kspace)

        # Iterate over each spoke
        for i in range(num_spokes):
            angle = self.traj[i]
            for r in torch.linspace(0, min(center_y, center_x), steps=min(center_y, center_x)):
                x = int(center_x + r * torch.cos(angle))
                y = int(center_y + r * torch.sin(angle))
                if 0 <= x < N_x and 0 <= y < N_y:
                    output[:, :, y, x] = kspace[:, :, y, x]

        return output
    
    def adj(self, input) -> torch.Tensor:
        """
        SENSE Adjoint Operator: from k-space to image
        """
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)

        kfull = torch.swapaxes(input, 0, 1)  # coils, slices, N_y, N_x

        if self.traj is None:
            # Cartesian sampling
            kmask = torch.swapaxes(self.mask * kfull, 0, 1)
            
        else:
            # Radial sampling
            kmask = self.radial_sample(kfull)

        imask = ifftc(kmask, norm='ortho')
        output = torch.sum(imask * self.coil.conj(), dim=1)  # slices, N_y, N_x

        return output

#Not verified yet
class SenseSp(): 
    def __init__(self, coil, mask, traj=None):
        """
        Implementation of the SENSE Operator based on SigPy.
        Args:
            coil: [N_batch, N_coil, N_y, N_x] - Coil sensitivity maps
            mask: [N_batch, N_y, N_x] - Sampling mask
            traj: [num_spokes, num_samples_per_spoke, 2] - Trajectory for radial sampling
        """
        self.coil = coil
        self.mask = mask
        self.traj = traj

    def fwd(self, input):
        """
        SENSE Forward Operator: from image to k-space
        """
        # Apply coil sensitivities
        img_coils = self.coil * input[:, None, :, :]  # Apply sensitivities to input image
        
        # Compute the FFT of the image for each coil
        kspace_coils = sp.fft(img_coils, axes=(-2, -1))
        
        # Apply the sampling mask
        if self.traj is None:
            # Cartesian sampling
            output = kspace_coils * self.mask[:, None, :, :]
        else:
            # Radial sampling (uses a predefined trajectory)
            output = mr.noncart.recon(kspace_coils, self.traj, img_shape=input.shape[-2:])

        return output

    def adj(self, input):
        """
        SENSE Adjoint Operator: from k-space to image
        """
        if self.traj is None:
            # Cartesian sampling
            masked_kspace = input * self.mask[:, None, :, :]
        else:
            # Handle radial sampling trajectory (if self.traj is not None)
            masked_kspace = input  # This should already be radially sampled

        # Compute the IFFT
        img_coils = sp.ifft(masked_kspace, axes=(-2, -1))
        
        # Combine images from all coils using the conjugate coil sensitivities
        combined_img = np.sum(img_coils * np.conj(self.coil), axis=1)
        
        return combined_img

