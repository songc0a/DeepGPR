import torch
import torch.nn as nn
import torch.nn.functional as F
import math

c = 299792458.0
m0 = 4.0 * math.pi * 1e-7
e0 = 1.0 / (m0 * c * c)

def _require_finite_tensor(name, tensor):
    """Validate that a tensor contains only finite values.

    Args:
        name: Name used in the error message.
        tensor: Tensor to check, or None to skip the check.
    """
    if tensor is not None and not torch.isfinite(tensor).all().item():
        raise ValueError(f"`{name}` contains NaN or Inf values.")


def initialization(device, er,se,mr,source_amplitudes,source_location,receiver_location,dx,dt,pmlthick):
    """Validate inputs and prepare model, source, receiver, and PML metadata.

    Args:
        device: PyTorch device where tensors will be stored.
        er: Relative permittivity tensor with shape (nx, ny) or (nx, ny, nz).
        se: Electrical conductivity tensor with the same shape as er.
        mr: Relative permeability tensor, or None to use ones.
        source_amplitudes: Source waveform tensor with shape (nwaveforms, nt, 1).
        source_location: Source coordinates with shape (nstep, nsr, 3).
        receiver_location: Receiver coordinates with shape (nstep, nrx, 3).
        dx: Spatial grid spacing.
        dt: Time step size.
        pmlthick: PML thickness as an int, list, or tensor.
    """
    dtype=torch.float32
    _require_finite_tensor("er", er)
    _require_finite_tensor("se", se)
    _require_finite_tensor("mr", mr)
    _require_finite_tensor("source_amplitudes", source_amplitudes)

    if er.min()<1 :
        raise ValueError('The values of epsilon is incorrect.(should be greater than 1)')
    if se.min()<0:
        raise ValueError('The values of sigma is incorrect.(should be non-negative)')
    
    if len(er.shape) == 2:
        er = er.reshape(*er.shape, 1)
    elif len(er.shape) != 3:
        raise ValueError('The shape of epsilon should be 2-d or 3-d.')
    
    if len(se.shape) == 2:
        se = se.reshape(*se.shape, 1)
    elif len(se.shape) != 3:
        raise ValueError('The shape of epsilon should be 2-d or 3-d.')

    if er.shape == se.shape:
        nx=er.shape[0]
        ny=er.shape[1]
        nz=er.shape[2]
        if nz==1:
            mode=2
        else:
            mode=3
        er=er.to(device)
        se=se.to(device)
        if mr is None:
            mr=torch.ones_like(er, device=device)
        else:
            if mr.shape == er.shape:
                mr=mr.to(device)
            else:
                raise ValueError('The shape of miu should be the same as epsilon and sigma.')
    else:
        raise ValueError('The shape of epsilon and sigma should be the same.')

    if source_location.shape[0] == receiver_location.shape[0]:
        source_location=source_location.to(torch.int)
        receiver_location=receiver_location.to(torch.int)

        source_check = (source_location >= 0).all()
        receiver_check = (receiver_location >= 0).all()

        source_check &= (source_location[..., 0] < nx).all()
        source_check &= (source_location[..., 1] < ny).all()
        source_check &= (source_location[..., 2] < nz).all()

        if not (source_check):
            raise ValueError(
                "Error: Source coordinates out of range! "
                f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
            )
        
        receiver_check &= (receiver_location[..., 0] < nx).all()
        receiver_check &= (receiver_location[..., 1] < ny).all()
        receiver_check &= (receiver_location[..., 2] < nz).all()

        if not (receiver_check):
            raise ValueError(
                "Error: Receiver coordinates out of range! "
                f"Valid ranges are x∈[0,{nx}), y∈[0,{ny}), z∈[0,{nz})"
            )
        nstep=source_location.shape[0]

        nsr=source_location.shape[1]
        nrx=receiver_location.shape[1]
        
        source_location=source_location.to(device)
        receiver_location=receiver_location.to(device)
    else:
        raise ValueError('The first dimension (nstep) of source_location and receiver_location should be the same.')
    
    source_amplitudes=source_amplitudes.to(device).contiguous()

    if (source_amplitudes.shape[0]>1 and source_amplitudes.shape[0]<nsr) or source_amplitudes.shape[0]>nsr :
        raise ValueError('The number of source waveforms is incorrect.')
    elif source_amplitudes.shape[0]==1 and nsr!=1:
        source_amplitudes=source_amplitudes.repeat(nsr,1,1).contiguous()
        print('Tips: The number of source waveforms is 1, but the number of sources is ',nsr,'. The source waveform is repeated for all sources.')

    check_cfl(dx, dt,nx,ny,nz)

    nt=source_amplitudes.shape[1]
    

    pmlthick=pmlthick_revert(pmlthick,er)
    ere=F.pad(er, (0, 1, 0, 1, 0, 1)).to(dtype)
    see=F.pad(se, (0, 1, 0, 1, 0, 1)).to(dtype)
    mr=F.pad(mr, (0, 1, 0, 1, 0, 1)).to(dtype)

    return er,se,nx,ny,nz,nt,nstep,nsr,nrx,ere,see,mr,mode,dtype,pmlthick,source_amplitudes


def check_cfl(dx, dt, nx,ny,nz):
    """Check the CFL stability condition for the simulation grid.

    Args:
        dx: Spatial grid spacing.
        dt: Time step size.
        nx: Number of cells along the x axis.
        ny: Number of cells along the y axis.
        nz: Number of cells along the z axis.
    """

    dy=dx
    dz=dx

    if nz==1:
        dt_max = 1.0 / (c * math.sqrt(1/dx**2 + 1/dy**2))
    elif nx==1:
        dt_max = 1.0 / (c * math.sqrt(1/dy**2 + 1/dz**2))
    elif ny==1:
        dt_max = 1.0 / (c * math.sqrt(1/dx**2 + 1/dz**2))
    else:
        dt_max = 1.0 / (c * math.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))

    if dt > dt_max:
        raise ValueError(f"Does not meet CFL conditions: dt={dt:.3e} > dt_max={dt_max:.3e}")


def pmlthick_revert(p, er):
    """Convert user PML thickness input to a six-boundary tensor.

    Args:
        p: PML thickness as an int, list, or tensor.
        er: Relative permittivity tensor used to detect 2D or 3D mode.
    """
    if isinstance(p, int):  
        if er.shape[2] == 1:
            return torch.tensor([p, p, p, p, 0, 0], dtype=torch.int32)
        return torch.tensor([p]*6, dtype=torch.int32)
    
    elif isinstance(p, list):
        if len(p) == 6:
            return torch.tensor(p, dtype=torch.int32)
        elif len(p) == 4:
            return torch.tensor(p + [0, 0], dtype=torch.int32)
        else:
            raise ValueError(f"Unsupported list length: {len(p)}. Must be 4 or 6.")
    
    elif isinstance(p, torch.Tensor):
        return p.to(dtype=torch.int32)
    
    else:
        raise TypeError(f"Unsupported type: {type(p)}")


class TVRegularization(nn.Module):
    """Total variation regularization for permittivity and conductivity models.

    Args:
        weight_ep: Weight applied to the permittivity TV loss.
        weight_sigma: Weight applied to the conductivity TV loss.
        method: TV variant, either "anisotropic" or another value for isotropic.
    """
    def __init__(self, weight_ep=1, weight_sigma=0.001, method='anisotropic'):
        """Create a TV regularization module.

        Args:
            weight_ep: Weight applied to the permittivity TV loss.
            weight_sigma: Weight applied to the conductivity TV loss.
            method: TV variant, either "anisotropic" or another value for isotropic.
        """
        super(TVRegularization, self).__init__()
        self.weight_ep = weight_ep
        self.weight_sigma = weight_sigma
        self.method = method

    def _compute_tv(self, data):
        """Compute total variation for one tensor.

        Args:
            data: 2D, 3D, or batched tensor to regularize.
        """
        if data.dim() == 3 and data.shape[-1] == 1:
            data = data.squeeze(-1)
            
        if data.dim() == 2:
            d_x = data[1:, :] - data[:-1, :]
            d_y = data[:, 1:] - data[:, :-1]
            
            loss_x = torch.sum(torch.abs(d_x))
            loss_y = torch.sum(torch.abs(d_y))
            loss_z = 0.0
            
        elif data.dim() == 3:
            d_x = data[1:, :, :] - data[:-1, :, :]
            d_y = data[:, 1:, :] - data[:, :-1, :]
            d_z = data[:, :, 1:] - data[:, :, :-1]
            
            loss_x = torch.sum(torch.abs(d_x))
            loss_y = torch.sum(torch.abs(d_y))
            loss_z = torch.sum(torch.abs(d_z))
            
        elif data.dim() == 4:
             d_x = data[..., 1:, :] - data[..., :-1, :]
             d_y = data[..., :, 1:] - data[..., :, :-1]
             
             loss_x = torch.sum(torch.abs(d_x))
             loss_y = torch.sum(torch.abs(d_y))
             loss_z = 0.0
             
        else:
            return torch.tensor(0.0, device=data.device)

        if self.method == 'anisotropic':

            total_tv = loss_x + loss_y + loss_z
        else:

            total_tv = (loss_x**2 + loss_y**2 + loss_z**2 + 1e-8).sqrt()

        return total_tv 

    def forward(self, ep=None, sigma=None):
        """Return the weighted TV loss for model parameters.

        Args:
            ep: Relative permittivity tensor, or None to skip it.
            sigma: Conductivity tensor, or None to skip it.
        """
        loss = torch.tensor(0.0, device=ep.device if ep is not None else sigma.device)

        if ep is not None and self.weight_ep > 0:
            loss += self.weight_ep * self._compute_tv(ep)

        if sigma is not None and self.weight_sigma > 0:
            loss += self.weight_sigma * self._compute_tv(sigma)
            
        return loss


def create_or_separate(tuple:tuple, nx,ny,nz,nstep,device: torch.device,
                  dtype: torch.dtype):
    """Create zero field components or validate existing field components.

    Args:
        tuple: Existing (x, y, z) field tensors, or None to allocate zeros.
        nx: Number of model cells along the x axis.
        ny: Number of model cells along the y axis.
        nz: Number of model cells along the z axis.
        nstep: Number of shots or simulations in the batch.
        device: PyTorch device for allocated tensors.
        dtype: PyTorch dtype for allocated tensors.
    """
    if tuple == None:
        return torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous(),torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous(),torch.zeros((nstep,nx+1,ny+1,nz+1), device=device, dtype=dtype).contiguous()
    # else:
    #     if tensor[0].shape[1]==nx+1 and tensor[0].shape[2]==ny+1 and tensor[0].shape[3]==nz+1 and tensor[0].shape[0]==nstep:
    #       return tensor[0].contiguous(),tensor[1].contiguous(),tensor[2].contiguous()
    #     else:
    #       print(nstep,nx,ny,nz)
    #       raise ValueError('The shape of E and H should be (nstep,nx+1,ny+1,nz+1).')
    else:
        condition = (
        tuple[0].shape[0] == nstep and
        tuple[0].shape[1] == nx + 1 and
        tuple[0].shape[2] == ny + 1
        )
    
        if tuple[0].ndim > 3:
            condition = condition and (tuple[0].shape[3] == nz + 1)
        
        if condition:
            return (
                tuple[0].contiguous(),
                tuple[1].contiguous(),
                tuple[2].contiguous()
            )
        else:
            actual_shape = list(tuple[0].shape)
            expected_min_shape = [nstep, nx + 1, ny + 1, nz + 1]
            raise ValueError(
                f"Shape not match! \n"
                f"Actual shape: {actual_shape} \n"
                f"Expected shape (at least): {expected_min_shape[:3]} and (if nz!=1) {nz+1}"
            )



def check_tensors_for_nan_inf(d,**tensors):
    """Check multiple named tensors for NaN or Inf values.

    Args:
        d: Label describing the current calculation stage.
        **tensors: Named tensors to validate.
    """
    found_issue = False

    for name, tensor in tensors.items():
        if tensor is None:
            print(f"[WARNING]{d}: {name} is None.")
            continue

        if not isinstance(tensor, torch.Tensor):
            print(f"[WARNING]{d}: {name} is not a tensor: {type(tensor)}")
            continue

        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan or has_inf:
            found_issue = True
            print(f"[ERROR]{d}: Tensor `{name}` contains:", end=" ")
            if has_nan:
                print("NaN ", end="")
            if has_inf:
                print("Inf ", end="")
            print(f"| shape={tuple(tensor.shape)} | dtype={tensor.dtype}")

    if found_issue:
        raise FloatingPointError(f"{d} produced tensors containing NaN or Inf values.")


def buildpmlcoeffs(er,mr,dt,dx,nx,ny,nz,pmlthick,device,dtype):
    """Build CPML region descriptors and update coefficients.

    Args:
        er: Relative permittivity tensor.
        mr: Relative permeability tensor.
        dt: Time step size.
        dx: Spatial grid spacing.
        nx: Number of model cells along the x axis.
        ny: Number of model cells along the y axis.
        nz: Number of model cells along the z axis.
        pmlthick: Six-boundary PML thickness tensor.
        device: PyTorch device for output tensors.
        dtype: PyTorch dtype for output tensors.
    """
    averageer=torch.zeros(6, device=device, dtype=dtype)
    averagemr=torch.zeros(6, device=device, dtype=dtype)
    lencfs=1
    x0 = torch.empty(0)
    xm = torch.empty(0)
    y0 = torch.empty(0)
    ym = torch.empty(0)
    z0 = torch.empty(0)
    zm = torch.empty(0)
    x01 = torch.empty(0)
    x02 = torch.empty(0)
    xm1 = torch.empty(0)
    xm2 = torch.empty(0)
    y01 = torch.empty(0)
    y02 = torch.empty(0)
    ym1 = torch.empty(0)
    ym2 = torch.empty(0)
    z01 = torch.empty(0)
    z02 = torch.empty(0)
    zm1 = torch.empty(0)
    zm2 = torch.empty(0)
    if pmlthick[0]>0:
        x0=torch.tensor((pmlthick[0],0,pmlthick[0],0,ny,0,nz), device=device, dtype=torch.int)
        averageer[0]=er[x0[1],:ny,:nz].mean()
        averagemr[0]=mr[x0[1],:ny,:nz].mean()
        CFS0=CFS(device=device)
        x01=torch.zeros((4,lencfs,pmlthick[0]), device=device, dtype=dtype)
        x02=torch.zeros((4,lencfs,pmlthick[0]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS0,x01,x02, averageer[0], averagemr[0], dt,dx,pmlthick[0])

    if pmlthick[1]>0:
        xm=torch.tensor((pmlthick[1],nx-pmlthick[1],nx,0,ny,0,nz), device=device, dtype=torch.int)
        averageer[1]=er[xm[1],:ny,:nz].mean()
        averagemr[1]=mr[xm[1],:ny,:nz].mean()
        CFS1=CFS(device=device)
        xm1=torch.zeros((4,lencfs,pmlthick[1]), device=device, dtype=dtype)
        xm2=torch.zeros((4,lencfs,pmlthick[1]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS1,xm1,xm2, averageer[1], averagemr[1], dt,dx,pmlthick[1])

    if pmlthick[2]>0:
        y0=torch.tensor((pmlthick[2],0,nx,0,pmlthick[2],0,nz), device=device, dtype=torch.int)
        averageer[2]=er[:nx,y0[3],:nz].mean()
        averagemr[2]=mr[:nx,y0[3],:nz].mean()
        CFS2=CFS(device=device)
        y01=torch.zeros((4,lencfs,pmlthick[2]), device=device, dtype=dtype)
        y02=torch.zeros((4,lencfs,pmlthick[2]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS2,y01,y02, averageer[2], averagemr[2], dt,dx,pmlthick[2])

    if pmlthick[3]>0:
        ym=torch.tensor((pmlthick[3],0,nx,ny-pmlthick[3],ny,0,nz), device=device, dtype=torch.int)
        averageer[3]=er[:nx,ym[3],:nz].mean()
        averagemr[3]=mr[:nx,ym[3],:nz].mean()
        CFS3=CFS(device=device)
        ym1=torch.zeros((4,lencfs,pmlthick[3]), device=device, dtype=dtype)
        ym2=torch.zeros((4,lencfs,pmlthick[3]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS3,ym1,ym2, averageer[3], averagemr[3], dt,dx,pmlthick[3])

    if pmlthick[4]>0:
        z0=torch.tensor((pmlthick[4],0,nx,0,ny,0,pmlthick[4]), device=device, dtype=torch.int)
        averageer[4]=er[:nx,:ny,z0[5]].mean()
        averagemr[4]=mr[:nx,:ny,z0[5]].mean()
        CFS4=CFS(device=device)
        z01=torch.zeros((4,lencfs,pmlthick[4]), device=device, dtype=dtype)
        z02=torch.zeros((4,lencfs,pmlthick[4]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS4,z01,z02, averageer[4], averagemr[4], dt,dx,pmlthick[4])

    if pmlthick[5]>0:
        zm=torch.tensor((pmlthick[5],0,nx,0,ny,nz-pmlthick[5],nz), device=device, dtype=torch.int)
        averageer[5]=er[:nx,:ny,zm[5]].mean()
        averagemr[5]=mr[:nx,:ny,zm[5]].mean()
        CFS5=CFS(device=device)
        zm1=torch.zeros((4,lencfs,pmlthick[5]), device=device, dtype=dtype)
        zm2=torch.zeros((4,lencfs,pmlthick[5]), device=device, dtype=dtype)
        calculate_pml_update_coeffs(CFS5,zm1,zm2, averageer[5], averagemr[5], dt,dx,pmlthick[5])
    return x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2



class CFSParameter(object):
    """Parameter settings for complex frequency shifted PML profiles.

    Args:
        ID: Parameter name, such as "alpha", "kappa", or "sigma".
        scaling: Scaling family used to generate the profile.
        scalingprofile: Profile order name used by polynomial scaling.
        min: Minimum value of the profile.
        max: Maximum value of the profile.
    """
    scalingprofiles = {'constant': 0, 'linear': 1, 'quadratic': 2, 'cubic': 3, 'quartic': 4, 'quintic': 5, 'sextic': 6, 'septic': 7, 'octic': 8}

    def __init__(self,ID =None, scaling='polynomial', scalingprofile=None, min=0, max=0):
        """Create one CFS profile parameter.

        Args:
            ID: Parameter name, such as "alpha", "kappa", or "sigma".
            scaling: Scaling family used to generate the profile.
            scalingprofile: Profile order name used by polynomial scaling.
            min: Minimum value of the profile.
            max: Maximum value of the profile.
        """
        self.ID = ID
        self.scaling = scaling
        self.scalingprofile = scalingprofile
        self.min = min
        self.max = max


class CFS(object):
    """Complex frequency shifted PML coefficient generator.

    Args:
        device: PyTorch device where generated profiles are stored.
    """

    def __init__(self, device):
        """Create a CFS parameter container.

        Args:
            device: PyTorch device where generated profiles are stored.
        """
        self.alpha = CFSParameter(ID='alpha', scalingprofile='constant')
        self.kappa = CFSParameter(ID='kappa', scalingprofile='constant', min=1, max=1)
        self.sigma = CFSParameter(ID='sigma', scalingprofile='quartic', min=0, max=None)
        self.device = device

    def calculate_sigmamax(self, d, er, mr):
        """Calculate the maximum PML sigma value.

        Args:
            d: Spatial grid spacing normal to the PML boundary.
            er: Average relative permittivity near the boundary.
            mr: Average relative permeability near the boundary.
        """
        with torch.no_grad():
            m = CFSParameter.scalingprofiles[self.sigma.scalingprofile]
            self.sigma.max = (0.8 * (m + 1)) / (((m0 / e0) ** 0.5) * d * torch.sqrt(er * mr))


    def scaling_polynomial(self, order, Evalues, Hvalues):
        """Create staggered electric and magnetic polynomial profiles.

        Args:
            order: Polynomial order index.
            Evalues: Electric profile tensor to fill.
            Hvalues: Magnetic profile tensor to fill.
        """
        tmp = (torch.linspace(0, (len(Evalues) - 1) + 0.5, steps=2 * len(Evalues)) / (len(Evalues) - 1)) ** order
        Evalues = tmp[0:-1:2].to(self.device)
        Hvalues = tmp[1::2].to(self.device)
        return Evalues, Hvalues

    def calculate_values(self, thickness, parameter):
        """Calculate electric and magnetic CFS values for one parameter.

        Args:
            thickness: PML thickness in grid cells.
            parameter: CFSParameter object to evaluate.
        """

        Evalues = torch.zeros(thickness + 1, device=self.device)
        Hvalues = torch.zeros(thickness + 1, device=self.device)
        if parameter.scalingprofile == 'constant':
            Evalues += parameter.max
            Hvalues += parameter.max
        elif parameter.scaling == 'polynomial':
            Evalues, Hvalues = self.scaling_polynomial(CFSParameter.scalingprofiles[parameter.scalingprofile], Evalues, Hvalues)
            if parameter.ID == 'alpha':
                Evalues = Evalues * (self.alpha.max - self.alpha.min) + self.alpha.min
                Hvalues = Hvalues * (self.alpha.max - self.alpha.min) + self.alpha.min
            elif parameter.ID == 'kappa':
                Evalues = Evalues * (self.kappa.max - self.kappa.min) + self.kappa.min
                Hvalues = Hvalues * (self.kappa.max - self.kappa.min) + self.kappa.min
            elif parameter.ID == 'sigma':
                Evalues = Evalues * (self.sigma.max - self.sigma.min) + self.sigma.min
                Hvalues = Hvalues * (self.sigma.max - self.sigma.min) + self.sigma.min

        Evalues = Evalues[:-1]
        Hvalues = Hvalues[:-1]
        
        return Evalues, Hvalues

def calculate_pml_update_coeffs(cfs,R1,R2, aver, avmr, dt,d,thickness):
    """Fill PML electric and magnetic update coefficient tensors.

    Args:
        cfs: CFS coefficient generator.
        R1: Electric-field PML coefficient tensor to fill.
        R2: Magnetic-field PML coefficient tensor to fill.
        aver: Average relative permittivity near the boundary.
        avmr: Average relative permeability near the boundary.
        dt: Time step size.
        d: Spatial grid spacing normal to the PML boundary.
        thickness: PML thickness in grid cells.
    """
    if not cfs.sigma.max:
        cfs.calculate_sigmamax(d, aver, avmr)

    Ealpha, Halpha = cfs.calculate_values(thickness, cfs.alpha)
    Ekappa, Hkappa = cfs.calculate_values(thickness, cfs.kappa)
    Esigma, Hsigma = cfs.calculate_values(thickness, cfs.sigma)

    R1=R1.contiguous()
    R2=R2.contiguous()

    tmp = (2 * e0 * Ekappa) + dt * (Ealpha * Ekappa + Esigma)
    R1[0,0, :] = (2 * e0 + dt * Ealpha) / tmp
    R1[1,0, :] = (2 * e0 * Ekappa) / tmp
    R1[2,0, :] = ((2 * e0 * Ekappa) - dt * (Ealpha * Ekappa + Esigma)) / tmp
    R1[3,0, :] = (2 * Esigma * dt) / (Ekappa * tmp)

    tmp = (2 * e0 * Hkappa) + dt * (Halpha * Hkappa + Hsigma)
    R2[0,0, :] = (2 * e0 + dt * Halpha) / tmp
    R2[1,0, :] = (2 * e0 * Hkappa) / tmp
    R2[2,0, :] = ((2 * e0 * Hkappa) - dt * (Halpha * Hkappa + Hsigma)) / tmp
    R2[3,0, :] = (2 * Hsigma * dt) / (Hkappa * tmp)



def build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device):
    """Create or reuse CPML auxiliary phi tensors.

    Args:
        x0: Descriptor for the low-x PML boundary.
        xm: Descriptor for the high-x PML boundary.
        y0: Descriptor for the low-y PML boundary.
        ym: Descriptor for the high-y PML boundary.
        z0: Descriptor for the low-z PML boundary.
        zm: Descriptor for the high-z PML boundary.
        nstep: Number of shots or simulations in the batch.
        PML: Existing tuple of 24 PML phi tensors, or None.
        device: PyTorch device for allocated tensors.
    """
   
    (x0EPhi1, x0EPhi2, x0HPhi1, x0HPhi2,
    xmEPhi1, xmEPhi2, xmHPhi1, xmHPhi2,
    y0EPhi1, y0EPhi2, y0HPhi1, y0HPhi2,
    ymEPhi1, ymEPhi2, ymHPhi1, ymHPhi2,
    z0EPhi1, z0EPhi2, z0HPhi1, z0HPhi2,
    zmEPhi1, zmEPhi2, zmHPhi1, zmHPhi2) = [torch.empty(0) for _ in range(24)]

    if PML==None:
        PML=(None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None)

    if x0.numel()!=0 and PML[0]==None:
        x0EPhi1=torch.zeros((nstep, int(x0[2]-x0[1]+1), int(x0[4]-x0[3]), int(x0[6]-x0[5]+1)),dtype=torch.float, device=device)
        x0EPhi2=torch.zeros((nstep, int(x0[2]-x0[1]+1), int(x0[4]-x0[3]+1), int(x0[6]-x0[5])),dtype=torch.float, device=device)
        x0HPhi1=torch.zeros((nstep, int(x0[2]-x0[1]), int(x0[4]-x0[3]+1), int(x0[6]-x0[5])), dtype=torch.float, device=device)
        x0HPhi2=torch.zeros((nstep, int(x0[2]-x0[1]), int(x0[4]-x0[3]), int(x0[6]-x0[5]+1)), dtype=torch.float, device=device)
    elif PML[0]!=None:
        x0EPhi1=PML[0].contiguous()
        x0EPhi2=PML[1].contiguous()
        x0HPhi1=PML[2].contiguous()
        x0HPhi2=PML[3].contiguous()

    if xm.numel()!=0 and PML[4]==None:
        xmEPhi1=torch.zeros((nstep, xm[2]-xm[1]+1, xm[4]-xm[3], xm[6]-xm[5]+1), dtype=torch.float, device=device)
        xmEPhi2=torch.zeros((nstep, xm[2]-xm[1]+1, xm[4]-xm[3]+1, xm[6]-xm[5]), dtype=torch.float, device=device)
        xmHPhi1=torch.zeros((nstep, xm[2]-xm[1], xm[4]-xm[3]+1, xm[6]-xm[5]), dtype=torch.float, device=device)
        xmHPhi2=torch.zeros((nstep, xm[2]-xm[1], xm[4]-xm[3], xm[6]-xm[5]+1), dtype=torch.float, device=device)    
    elif PML[4]!=None:
        xmEPhi1=PML[4].contiguous()
        xmEPhi2=PML[5].contiguous()
        xmHPhi1=PML[6].contiguous()
        xmHPhi2=PML[7].contiguous()

    if y0.numel()!=0 and PML[8]==None:
        y0EPhi1=torch.zeros((nstep, y0[2]-y0[1], y0[4]-y0[3]+1, y0[6]-y0[5]+1), dtype=torch.float, device=device)
        y0EPhi2=torch.zeros((nstep, y0[2]-y0[1]+1, y0[4]-y0[3]+1, y0[6]-y0[5]), dtype=torch.float, device=device)
        y0HPhi1=torch.zeros((nstep, y0[2]-y0[1]+1, y0[4]-y0[3], y0[6]-y0[5]), dtype=torch.float, device=device)
        y0HPhi2=torch.zeros((nstep, y0[2]-y0[1], y0[4]-y0[3], y0[6]-y0[5]+1), dtype=torch.float, device=device)
    elif PML[8]!=None:
        y0EPhi1=PML[8].contiguous()
        y0EPhi2=PML[9].contiguous()
        y0HPhi1=PML[10].contiguous()
        y0HPhi2=PML[11].contiguous()

    if ym.numel()!=0 and PML[12]==None:
        ymEPhi1=torch.zeros((nstep, ym[2]-ym[1], ym[4]-ym[3]+1, ym[6]-ym[5]+1),dtype=torch.float, device=device)
        ymEPhi2=torch.zeros((nstep, ym[2]-ym[1]+1, ym[4]-ym[3]+1, ym[6]-ym[5]), dtype=torch.float, device=device)
        ymHPhi1=torch.zeros((nstep, ym[2]-ym[1]+1, ym[4]-ym[3], ym[6]-ym[5]), dtype=torch.float, device=device)
        ymHPhi2=torch.zeros((nstep, ym[2]-ym[1], ym[4]-ym[3], ym[6]-ym[5]+1), dtype=torch.float, device=device)
    elif PML[12]!=None:
        ymEPhi1=PML[12].contiguous()
        ymEPhi2=PML[13].contiguous()
        ymHPhi1=PML[14].contiguous()
        ymHPhi2=PML[15].contiguous()

    if z0.numel()!=0 and PML[16]==None:
        z0EPhi1=torch.zeros((nstep, z0[2]-z0[1], z0[4]-z0[3]+1, z0[6]-z0[5]+1), dtype=torch.float, device=device)
        z0EPhi2=torch.zeros((nstep, z0[2]-z0[1]+1, z0[4]-z0[3], z0[6]-z0[5]+1), dtype=torch.float, device=device)
        z0HPhi1=torch.zeros((nstep, z0[2]-z0[1]+1, z0[4]-z0[3], z0[6]-z0[5]), dtype=torch.float, device=device)
        z0HPhi2=torch.zeros((nstep, z0[2]-z0[1], z0[4]-z0[3]+1, z0[6]-z0[5]), dtype=torch.float, device=device)
    elif PML[16]!=None:
        z0EPhi1=PML[16].contiguous()
        z0EPhi2=PML[17].contiguous()
        z0HPhi1=PML[18].contiguous()
        z0HPhi2=PML[19].contiguous()

    if zm.numel()!=0 and PML[20]==None:
        zmEPhi1=torch.zeros((nstep, zm[2]-zm[1], zm[4]-zm[3]+1, zm[6]-zm[5]+1), dtype=torch.float, device=device)
        zmEPhi2=torch.zeros((nstep, zm[2]-zm[1]+1, zm[4]-zm[3], zm[6]-zm[5]+1), dtype=torch.float, device=device)
        zmHPhi1=torch.zeros((nstep, zm[2]-zm[1]+1, zm[4]-zm[3], zm[6]-zm[5]), dtype=torch.float, device=device)
        zmHPhi2=torch.zeros((nstep, zm[2]-zm[1], zm[4]-zm[3]+1, zm[6]-zm[5]), dtype=torch.float, device=device)
    elif PML[20]!=None:
        zmEPhi1=PML[20].contiguous()
        zmEPhi2=PML[21].contiguous()
        zmHPhi1=PML[22].contiguous()
        zmHPhi2=PML[23].contiguous()

    return x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2


def checkpoint_initial_field(device=None,per_nstep=None, dx=None, dt=None, 
            source_amplitudes=None,
            source_location=None, 
            receiver_location=None, 
            er=None, se=None,mr=None, 
            pmlthick=10):
    """Create initial electric, magnetic, and PML field checkpoints.

    Args:
        device: PyTorch device where tensors will be allocated.
        per_nstep: Optional number of shots to keep from the initialized fields.
        dx: Spatial grid spacing.
        dt: Time step size.
        source_amplitudes: Source waveform tensor.
        source_location: Source coordinates with shape (nstep, nsr, 3).
        receiver_location: Receiver coordinates with shape (nstep, nrx, 3).
        er: Relative permittivity tensor.
        se: Electrical conductivity tensor.
        mr: Relative permeability tensor, or None to use ones.
        pmlthick: PML thickness as an int, list, or tensor.
    """
    E=None
    H=None
    PML=None

    er,se,nx,ny,nz,_,nstep,_,_,_,_,mr,_,dtype,pmlthick,source_amplitudes=initialization(device,er,se,mr,source_amplitudes,source_location,receiver_location,dx,dt,pmlthick)

    Ex,Ey,Ez=create_or_separate(E,nx,ny,nz,nstep,device,dtype)
    Hx,Hy,Hz=create_or_separate(H,nx,ny,nz,nstep,device,dtype)

    x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2=buildpmlcoeffs(er,mr,dt,dx,nx,ny,nz,pmlthick,device,dtype)


    x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2=build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device)

    del x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2

    print("per_nstep:"+str(per_nstep))
    print("total step:"+str(nstep))
    # print_field_shapes((Ex,Ey,Ez),(Hx,Hy,Hz),(x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2))

    if per_nstep==None:
        return (Ex,Ey,Ez),(Hx,Hy,Hz),(x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2)
    elif er.shape[2]==1:
        return (Ex[:per_nstep,:,:,:],Ey[:per_nstep,:,:,:],Ez[:per_nstep,:,:,:]),(Hx[:per_nstep,:,:,:],Hy[:per_nstep,:,:,:],Hz[:per_nstep,:,:,:]),(x0EPhi1[:per_nstep,:,:,:],x0EPhi2[:per_nstep,:,:,:],x0HPhi1[:per_nstep,:,:,:],x0HPhi2[:per_nstep,:,:,:],xmEPhi1[:per_nstep,:,:,:],xmEPhi2[:per_nstep,:,:,:],xmHPhi1[:per_nstep,:,:,:],xmHPhi2[:per_nstep,:,:,:],y0EPhi1[:per_nstep,:,:,:],y0EPhi2[:per_nstep,:,:,:],y0HPhi1[:per_nstep,:,:,:],y0HPhi2[:per_nstep,:,:,:],ymEPhi1[:per_nstep,:,:,:],ymEPhi2[:per_nstep,:,:,:],ymHPhi1[:per_nstep,:,:,:],ymHPhi2[:per_nstep,:,:,:],z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2)
    else:
        return (Ex[:per_nstep,:,:,:],Ey[:per_nstep,:,:,:],Ez[:per_nstep,:,:,:]),(Hx[:per_nstep,:,:,:],Hy[:per_nstep,:,:,:],Hz[:per_nstep,:,:,:]),(x0EPhi1[:per_nstep,:,:,:],x0EPhi2[:per_nstep,:,:,:],x0HPhi1[:per_nstep,:,:,:],x0HPhi2[:per_nstep,:,:,:],xmEPhi1[:per_nstep,:,:,:],xmEPhi2[:per_nstep,:,:,:],xmHPhi1[:per_nstep,:,:,:],xmHPhi2[:per_nstep,:,:,:],y0EPhi1[:per_nstep,:,:,:],y0EPhi2[:per_nstep,:,:,:],y0HPhi1[:per_nstep,:,:,:],y0HPhi2[:per_nstep,:,:,:],ymEPhi1[:per_nstep,:,:,:],ymEPhi2[:per_nstep,:,:,:],ymHPhi1[:per_nstep,:,:,:],ymHPhi2[:per_nstep,:,:,:],z0EPhi1[:per_nstep,:,:,:],z0EPhi2[:per_nstep,:,:,:],z0HPhi1[:per_nstep,:,:,:],z0HPhi2[:per_nstep,:,:,:],zmEPhi1[:per_nstep,:,:,:],zmEPhi2[:per_nstep,:,:,:],zmHPhi1[:per_nstep,:,:,:],zmHPhi2[:per_nstep,:,:,:])


def zero_field(*tensors):
    """Return zero tensors with the same shapes as the input tensors.

    Args:
        *tensors: Tensors to zero, or None values to preserve.
    """
    return tuple(torch.zeros_like(t) if t is not None else None for t in tensors)
