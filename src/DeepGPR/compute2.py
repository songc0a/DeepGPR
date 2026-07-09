import torch
import ctypes
from . import get_deepgpr_lib, set_library_fdtd_order
from .common import initialization,build_pml_phi,create_or_separate,buildpmlcoeffs,check_tensors_for_nan_inf


def _sync_cuda_device(device):
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)

def compute(device, dx=None, dt=None, 
            source_amplitudes=None,
            source_location=None, 
            receiver_location=None, 
            er=None, se=None,mr=None, 
            E=None,H=None,
            PML=None,
            pmlthick=10, source_direction=2, reciever_direction=2,
            model_gradient_sampling_interval=1,
            use_async_offload=False,
            fdtd_order=2):
    device = torch.device(device)
    if fdtd_order not in (2, 4, 8):
        raise ValueError("fdtd_order must be one of 2, 4, or 8.")

    er,se,nx,ny,nz,nt,nstep,nsr,nrx,ere,see,mr,mode,dtype,pmlthick,source_amplitudes=initialization(device,er,se,mr,source_amplitudes,source_location,receiver_location,dx,dt,pmlthick)

    Ex,Ey,Ez=create_or_separate(E,nx,ny,nz,nstep,device,dtype)
    Hx,Hy,Hz=create_or_separate(H,nx,ny,nz,nstep,device,dtype)

    x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2=buildpmlcoeffs(er,mr,dt,dx,nx,ny,nz,pmlthick,device,dtype)

    x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2=build_pml_phi(x0,xm,y0,ym,z0,zm,nstep,PML,device)

    Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2,Eall,receiver_amplitudes = DeepGPR.apply(
        er, se,Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2, mr,dx,nx,ny,nz,dt,nt,nstep,source_amplitudes,source_location,receiver_location,pmlthick,nsr,nrx,device,dtype,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see,source_direction, reciever_direction, model_gradient_sampling_interval, use_async_offload, fdtd_order)

    return Eall,(Ex,Ey,Ez),(Hx,Hy,Hz),(x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2),receiver_amplitudes


class DeepGPR(torch.autograd.Function):
    @staticmethod
    def forward(ctx, er, se,Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2, mr,dx,nx,ny,nz,dt,
                nt,nstep,source_amplitudes,source_location,receiver_location,
                pmlthick,nsr,nrx,device,dtype,x0,xm,
                y0,ym,z0,zm,x01,x02,xm1,xm2,
                y01,y02,ym1,ym2,z01,z02,zm1,zm2,
                ere,see,source_direction, reciever_direction, 
                model_gradient_sampling_interval, use_async_offload, fdtd_order):
        
        source_amplitudes = source_amplitudes.contiguous()
        source_location=source_location.to(torch.int32).contiguous()
        receiver_location=receiver_location.to(torch.int32).contiguous()
        c_lib = get_deepgpr_lib(device)
        set_library_fdtd_order(c_lib, fdtd_order)
        ctx.save_for_backward(er, se, mr,receiver_location,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see)
        ctx.dx=dx
        ctx.nx=nx
        ctx.ny=ny
        ctx.nz=nz
        ctx.dt=dt
        ctx.nt=nt
        ctx.nrx=nrx
        ctx.nsr=nsr
        ctx.nstep=nstep
        ctx.pmlthick=pmlthick
        ctx.device=device
        ctx.dtype=dtype
        ctx.model_gradient_sampling_interval = model_gradient_sampling_interval
        ctx.use_async_offload = bool(use_async_offload and device.type == "cuda")
        ctx.fdtd_order = fdtd_order

        nt_saved = (nt + model_gradient_sampling_interval - 1) // model_gradient_sampling_interval
        
        if ctx.use_async_offload:
            Eall = torch.zeros((nt_saved, nstep, nx, ny, nz), device='cpu', dtype=dtype).pin_memory()
        else:
            Eall = torch.zeros((nt_saved, nstep, nx, ny, nz), device=device, dtype=dtype).contiguous()

        Eupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)

        receiver_amplitudes = torch.zeros((nstep, 6, nt, nrx),device=device, dtype=dtype).contiguous()
        pml = [int(pmlthick[i]) for i in range(6)]

        _sync_cuda_device(device)
        c_lib.forward(
                ctypes.cast(ere.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(see.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(mr.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Eall.data_ptr(), ctypes.POINTER(ctypes.c_float)),              
                ctypes.cast(Ex.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Ey.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Ez.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hz.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                ctypes.cast(Eupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Eupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Eupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                ctypes.cast(x0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(x0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(x0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(x0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(xmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(xmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(y0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(y0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ymEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ymEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(ymHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ymHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(z0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(z0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(z0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(z0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(zmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(zmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                pml[0],pml[1],pml[2],
                pml[3],pml[4],pml[5],

                ctypes.cast(x01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym1.data_ptr(), ctypes.POINTER(ctypes.c_float)),       
                ctypes.cast(z01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),                
                ctypes.cast(x02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm2.data_ptr(), ctypes.POINTER(ctypes.c_float)),               
                ctypes.cast(y02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym2.data_ptr(), ctypes.POINTER(ctypes.c_float)),         
                ctypes.cast(z02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm2.data_ptr(), ctypes.POINTER(ctypes.c_float)),

                dt, nt, nstep, nrx, dx,
                ctypes.cast(receiver_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(receiver_amplitudes.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                nx+1, ny+1, nz+1, nsr,
                ctypes.cast(source_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(source_amplitudes.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                source_direction,
                model_gradient_sampling_interval)
        _sync_cuda_device(device)

        check_tensors_for_nan_inf(d="forward",
            Ex=Ex, Ey=Ey, Ez=Ez,
            Hx=Hx, Hy=Hy, Hz=Hz,
            x0EPhi1=x0EPhi1, x0EPhi2=x0EPhi2,
            x0HPhi1=x0HPhi1, x0HPhi2=x0HPhi2,
            xmEPhi1=xmEPhi1, xmEPhi2=xmEPhi2,
            xmHPhi1=xmHPhi1, xmHPhi2=xmHPhi2,
            y0EPhi1=y0EPhi1, y0EPhi2=y0EPhi2,
            y0HPhi1=y0HPhi1, y0HPhi2=y0HPhi2,
            ymEPhi1=ymEPhi1, ymEPhi2=ymEPhi2,
            ymHPhi1=ymHPhi1, ymHPhi2=ymHPhi2,
            z0EPhi1=z0EPhi1, z0EPhi2=z0EPhi2,
            z0HPhi1=z0HPhi1, z0HPhi2=z0HPhi2,
            zmEPhi1=zmEPhi1, zmEPhi2=zmEPhi2,
            zmHPhi1=zmHPhi1, zmHPhi2=zmHPhi2
        )

        ctx.Eall = Eall
        return (Ex,Ey,Ez,Hx,Hy,Hz,x0EPhi1,x0EPhi2,x0HPhi1,x0HPhi2,xmEPhi1,xmEPhi2,xmHPhi1,xmHPhi2,y0EPhi1,y0EPhi2,y0HPhi1,y0HPhi2,ymEPhi1,ymEPhi2,ymHPhi1,ymHPhi2,z0EPhi1,z0EPhi2,z0HPhi1,z0HPhi2,zmEPhi1,zmEPhi2,zmHPhi1,zmHPhi2,Eall,receiver_amplitudes[:,reciever_direction,:,:])

    @staticmethod
    def backward(ctx,gEx,gEy,gEz,gHx,gHy,gHz,gx0EPhi1,gx0EPhi2,gx0HPhi1,gx0HPhi2,gxmEPhi1,gxmEPhi2,gxmHPhi1,gxmHPhi2,gy0EPhi1,gy0EPhi2,gy0HPhi1,gy0HPhi2,gymEPhi1,gymEPhi2,gymHPhi1,gymHPhi2,gz0EPhi1,gz0EPhi2,gz0HPhi1,gz0HPhi2,gzmEPhi1,gzmEPhi2,gzmHPhi1,gzmHPhi2,gEall,gezreciver):
        
        sourceamp=gezreciver
        er, se, mr,receiver_location,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see=ctx.saved_tensors
        
        ere=ere.contiguous()
        see=see.contiguous()
        er=er.contiguous()
        se=se.contiguous()
        mr=mr.contiguous()
        receiver_location=receiver_location.contiguous()
        x0=x0.contiguous()
        xm=xm.contiguous()
        y0=y0.contiguous()
        ym=ym.contiguous()
        z0=z0.contiguous()
        zm=zm.contiguous()
        x01=x01.contiguous()
        x02=x02.contiguous()
        xm1=xm1.contiguous()
        xm2=xm2.contiguous()
        y01=y01.contiguous()
        y02=y02.contiguous()
        ym1=ym1.contiguous()
        ym2=ym2.contiguous()
        z01=z01.contiguous()
        z02=z02.contiguous()
        zm1=zm1.contiguous()
        zm2=zm2.contiguous()

        dx=ctx.dx
        nx=ctx.nx
        ny=ctx.ny
        nz=ctx.nz
        dt=ctx.dt
        nt=ctx.nt
        nsr=ctx.nrx
        nrx=ctx.nsr
        dtype=ctx.dtype
        nstep=ctx.nstep
        pmlthick=ctx.pmlthick
        device=ctx.device
        Eall=ctx.Eall
        model_gradient_sampling_interval = ctx.model_gradient_sampling_interval
        c_lib = get_deepgpr_lib(device)
        set_library_fdtd_order(c_lib, ctx.fdtd_order)

        Eall=Eall.contiguous()
        gEx=gEx.contiguous()
        gEy=gEy.contiguous()
        gEz=gEz.contiguous()
        gHx=gHx.contiguous()
        gHy=gHy.contiguous()
        gHz=gHz.contiguous()
        gx0EPhi1=gx0EPhi1.contiguous()
        gx0EPhi2=gx0EPhi2.contiguous()
        gx0HPhi1=gx0HPhi1.contiguous()
        gx0HPhi2=gx0HPhi2.contiguous()
        gxmEPhi1=gxmEPhi1.contiguous()
        gxmEPhi2=gxmEPhi2.contiguous()
        gxmHPhi1=gxmHPhi1.contiguous()
        gxmHPhi2=gxmHPhi2.contiguous()
        gy0EPhi1=gy0EPhi1.contiguous()
        gy0EPhi2=gy0EPhi2.contiguous()
        gy0HPhi1=gy0HPhi1.contiguous()
        gy0HPhi2=gy0HPhi2.contiguous()
        gymEPhi1=gymEPhi1.contiguous()
        gymEPhi2=gymEPhi2.contiguous()
        gymHPhi1=gymHPhi1.contiguous()
        gymHPhi2=gymHPhi2.contiguous()
        gz0EPhi1=gz0EPhi1.contiguous()
        gz0EPhi2=gz0EPhi2.contiguous()
        gz0HPhi1=gz0HPhi1.contiguous()
        gz0HPhi2=gz0HPhi2.contiguous()
        gzmEPhi1=gzmEPhi1.contiguous()
        gzmEPhi2=gzmEPhi2.contiguous()
        gzmHPhi1=gzmHPhi1.contiguous()
        gzmHPhi2=gzmHPhi2.contiguous()
        sourceamp=sourceamp.contiguous()

        Eupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Eupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs0=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs1=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)
        Hupdatecoffs4=torch.zeros((nx+1,ny+1,nz+1), device=device, dtype=dtype)

        if er.requires_grad:
            grad_er=torch.zeros((nx,ny,nz),device=device,dtype=dtype).contiguous()
            errequiregrad=1
        else:
            grad_er=torch.empty(0)
            errequiregrad=0

        if se.requires_grad: 
            grad_se=torch.zeros((nx,ny,nz),device=device,dtype=dtype).contiguous()
            serequiregrad=1
        else:
            grad_se=torch.empty(0)
            serequiregrad=0

        pml = [int(pmlthick[i]) for i in range(6)]

        _sync_cuda_device(device)
        c_lib.backward(
                ctypes.cast(ere.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(see.data_ptr(), ctypes.POINTER(ctypes.c_float)), 
                ctypes.cast(mr.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Eall.data_ptr(), ctypes.POINTER(ctypes.c_float)),               
                ctypes.cast(gEx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gEy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gEz.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gHx.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gHy.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gHz.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                ctypes.cast(Eupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Eupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Eupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs0.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(Hupdatecoffs1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(Hupdatecoffs4.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                ctypes.cast(gx0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gx0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gx0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gx0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gxmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gxmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gxmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gxmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gy0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gy0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gy0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gy0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gymEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gymEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gymHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gymHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gz0EPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gz0EPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gz0HPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gz0HPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gzmEPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gzmEPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(gzmHPhi1.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(gzmHPhi2.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                pml[0],pml[1],pml[2],
                pml[3],pml[4],pml[5],

                ctypes.cast(x01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                ctypes.cast(y01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym1.data_ptr(), ctypes.POINTER(ctypes.c_float)),       
                ctypes.cast(z01.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm1.data_ptr(), ctypes.POINTER(ctypes.c_float)),                
                ctypes.cast(x02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(xm2.data_ptr(), ctypes.POINTER(ctypes.c_float)),               
                ctypes.cast(y02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(ym2.data_ptr(), ctypes.POINTER(ctypes.c_float)),         
                ctypes.cast(z02.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(zm2.data_ptr(), ctypes.POINTER(ctypes.c_float)), 

                dt, nt, nstep, nrx, dx,
                nx+1, ny+1, nz+1, nsr,
                ctypes.cast(receiver_location.data_ptr(), ctypes.POINTER(ctypes.c_int)), ctypes.cast(sourceamp.data_ptr(), ctypes.POINTER(ctypes.c_float)),
                2,
                ctypes.cast(grad_er.data_ptr(), ctypes.POINTER(ctypes.c_float)), ctypes.cast(grad_se.data_ptr(), ctypes.POINTER(ctypes.c_float)),errequiregrad,serequiregrad,
                model_gradient_sampling_interval)  
        _sync_cuda_device(device)
        
        check_tensors_for_nan_inf(d="backward",
            gEx=gEx, gEy=gEy, gEz=gEz,
            gHx=gHx, gHy=gHy, gHz=gHz,
            gx0EPhi1=gx0EPhi1, gx0EPhi2=gx0EPhi2,
            gx0HPhi1=gx0HPhi1, gx0HPhi2=gx0HPhi2,
            gxmEPhi1=gxmEPhi1, gxmEPhi2=gxmEPhi2,
            gxmHPhi1=gxmHPhi1, gxmHPhi2=gxmHPhi2,
            gy0EPhi1=gy0EPhi1, gy0EPhi2=gy0EPhi2,
            gy0HPhi1=gy0HPhi1, gy0HPhi2=gy0HPhi2,
            gymEPhi1=gymEPhi1, gymEPhi2=gymEPhi2,
            gymHPhi1=gymHPhi1, gymHPhi2=gymHPhi2,
            gz0EPhi1=gz0EPhi1, gz0EPhi2=gz0EPhi2,
            gz0HPhi1=gz0HPhi1, gz0HPhi2=gz0HPhi2,
            gzmEPhi1=gzmEPhi1, gzmEPhi2=gzmEPhi2,
            gzmHPhi1=gzmHPhi1, gzmHPhi2=gzmHPhi2
        )

        ctx.Eall = None
        del Eall,er, se, mr,receiver_location,x0,xm,y0,ym,z0,zm,x01,x02,xm1,xm2,y01,y02,ym1,ym2,z01,z02,zm1,zm2,ere,see, Eupdatecoffs0, Eupdatecoffs1, Eupdatecoffs4, Hupdatecoffs0, Hupdatecoffs1, Hupdatecoffs4

        return (
                    grad_er, grad_se,         
                    gEx,gEy,gEz, gHx,gHy,gHz, 
                    gx0EPhi1,gx0EPhi2,gx0HPhi1,gx0HPhi2,
                    gxmEPhi1,gxmEPhi2,gxmHPhi1,gxmHPhi2,
                    gy0EPhi1,gy0EPhi2,gy0HPhi1,gy0HPhi2,
                    gymEPhi1,gymEPhi2,gymHPhi1,gymHPhi2,
                    gz0EPhi1,gz0EPhi2,gz0HPhi1,gz0HPhi2,
                    gzmEPhi1,gzmEPhi2,gzmHPhi1,gzmHPhi2,   
                    None, None, None, None, None, None, None, None,   
                    None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, 
                    None, None, None, None, None, None, None, None, 
                    None, None
                )
