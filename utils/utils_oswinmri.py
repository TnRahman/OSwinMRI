import torch
from torch import nn
import os
import cv2
import gc
import numpy as np
from scipy.io import *
from scipy.fftpack import *
from typing import Optional, Tuple



def fft2t(
    data: torch.Tensor,
    dim: Tuple[int, int] = (1, 2),
    centered: bool = True,
    normalized: bool = True
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ('height', 'width').
    centered: bool
        Whether to apply a centered fft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the fft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.

    Returns
    -------
    output_data: torch.Tensor
        The Fast Fourier transform of the data.
    """

    if centered:
        data = torch.fft.ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    
    data = torch.fft.fftn(
        data,
        dim=dim,
        norm="ortho" if normalized else None,
    )

    if centered:
        data = torch.fft.fftshift(data, dim=dim)

    return data


def ifft2t(
    data: torch.Tensor,
    dim: Tuple[int, int] = (1, 2),
    centered: bool = True,
    normalized: bool = True
) -> torch.Tensor:
    """Apply centered two-dimensional Inverse Fast Fourier Transform. Can be performed in half precision when input
    shapes are powers of two.

    Version for PyTorch >= 1.7.0.

    Parameters
    ----------
    data: torch.Tensor
        Complex-valued input tensor. Should be of shape (\*, 2) and dim is in \*.
    dim: tuple, list or int
        Dimensions over which to compute. Should be positive. Negative indexing not supported
        Default is (1, 2), corresponding to ( 'height', 'width').
    centered: bool
        Whether to apply a centered ifft (center of kspace is in the center versus in the corners).
        For FastMRI dataset this has to be true and for the Calgary-Campinas dataset false.
    normalized: bool
        Whether to normalize the ifft. For the FastMRI this has to be true and for the Calgary-Campinas dataset false.

    Returns
    -------
    output_data: torch.Tensor
        The Inverse Fast Fourier transform of the data.
    """

    if centered:
        data = torch.fft.ifftshift(data, dim=dim)
    # Verify whether half precision and if fft is possible in this shape. Else do a typecast.
    data = torch.fft.ifftn(
        data,
        dim=dim,
        norm="ortho" if normalized else None,
    )

    if centered:
        data = torch.fft.fftshift(data, dim=dim)

    return data

def fft2n(
    data,
    dim=(1,2),
    centered=True,
    normalized=True):

    if centered:
        data = np.fft.ifftshift(data, axes=dim)
    
    
    data = np.fft.fftn(
        data,
        axes=dim,
        norm="ortho" if normalized else None,
    )

    if centered:
        data = np.fft.fftshift(data, axes=dim)

    return data


def ifft2n(
    data,
    dim=(1,2),
    centered=True,
    normalized=True):

    if centered:
        data = np.fft.ifftshift(data, axes=dim)
    
    data = np.fft.ifftn(
        data,
        axes=dim,
        norm="ortho" if normalized else None,
    )

    if centered:
        data = np.fft.fftshift(data, axes=dim)

    return data



def data_consistency_unrolled(E_out_img_cs, Uks_cs, mask,  dcw, step=None, dc_step=None):

    E_ks_c = fft2t(E_out_img_cs[:,::2,:,:]+1j*E_out_img_cs[:,1::2,:,:],dim=(-2,-1),centered=True,normalized=False)

    E_ksm_c = torch.mul(E_ks_c , mask)
    EU_ksm_c = torch.mul(E_ks_c,1-mask)

    DC_ks = dcw(Uks_cs[:,::2,:,:]+1j*Uks_cs[:,1::2,:,:],EU_ksm_c)

    E_ks_dc_c = torch.add(E_ksm_c,DC_ks)
    
    E_img_dc_c = ifft2t(E_ks_dc_c,dim=(-2,-1),centered=True,normalized=False)

    E_img_dc_cs = torch.zeros_like(E_out_img_cs)
    E_img_dc_cs[:,::2,:,:] = E_img_dc_c.real
    E_img_dc_cs[:,1::2,:,:] = E_img_dc_c.imag
    
    return E_img_dc_cs


def data_consistency_fastmri(E_out_img_cs, Uks_cs, mask,  step=None, dc_step=None):
    # fft2t(self.H_c,dim=(-2,-1),centered=True,normalized=True)

    E_ks_c = fft2t(E_out_img_cs[:,::2,:,:]+1j*E_out_img_cs[:,1::2,:,:],dim=(-2,-1),centered=True,normalized=False)

    E_ksm_c = torch.mul(E_ks_c , mask)

    if step is not None:
        scf_dc = np.clip((dc_step-step)/dc_step,0,1)
        Es_comp_ksm_c = torch.mul(scf_dc,torch.mul(E_ks_c,1-mask))
        Ukss_c = torch.mul(1-scf_dc,Uks_cs[:,::2,:,:]+1j*Uks_cs[:,1::2,:,:])
        E_ks_dc_c = torch.add(E_ksm_c , torch.add(Es_comp_ksm_c,Ukss_c))
    else:
        E_ks_dc_c = torch.add(E_ksm_c , Uks_cs[:,::2,:,:]+1j*Uks_cs[:,1::2,:,:])

    E_img_dc_c = ifft2t(E_ks_dc_c,dim=(-2,-1),centered=True,normalized=False)

    E_img_dc_cs = torch.zeros_like(E_out_img_cs)
    E_img_dc_cs[:,::2,:,:] = E_img_dc_c.real
    E_img_dc_cs[:,1::2,:,:] = E_img_dc_c.imag
    
    return E_img_dc_cs


def data_consistency(E_out_img_cs, Uks_cs, mask,  step=None, dc_step=None):

    E_ks_c = torch.fft.fftshift(torch.fft.fftn(E_out_img_cs[:,::2,:,:]+1j*E_out_img_cs[:,1::2,:,:],dim=(-2,-1)),dim=(-2,-1))

    E_ksm_c = torch.mul(E_ks_c , mask)

    if step is not None:
        scf_dc = np.clip((dc_step-step)/dc_step,0,1)
        Es_comp_ksm_c = torch.mul(scf_dc,torch.mul(E_ks_c,1-mask))
        Ukss_c = torch.mul(1-scf_dc,Uks_cs[:,::2,:,:]+1j*Uks_cs[:,1::2,:,:])
        E_ks_dc_c = torch.add(E_ksm_c , torch.add(Es_comp_ksm_c,Ukss_c))
    else:
        E_ks_dc_c = torch.add(E_ksm_c , Uks_cs[:,::2,:,:]+1j*Uks_cs[:,1::2,:,:])

    E_img_dc_c = torch.fft.ifftn(torch.fft.ifftshift(E_ks_dc_c,dim=(-2,-1)),dim=(-2,-1))

    E_img_dc_cs = torch.zeros_like(E_out_img_cs)
    E_img_dc_cs[:,::2,:,:] = E_img_dc_c.real
    E_img_dc_cs[:,1::2,:,:] = E_img_dc_c.imag
    
    return E_img_dc_cs



# N-dimensional FFT (considering ifftshift/fftshift operations)
def fftnc(x, axes=(0, 1)):
    for ax in axes:
        x = 1 / np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

def ifftnc(x, axes=(0, 1)):
    # return np.sqrt(np.prod(x.shape[axes])) * np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)
    for ax in axes:
        x = np.sqrt(x.shape[ax]) * np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axes=ax), axis=ax), axes=ax)
    return x

