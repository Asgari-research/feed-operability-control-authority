#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optional non-isothermal cyclic model used for independent validation of the P10 analysis."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Callable, Dict, Any, Optional, Sequence, Tuple, List
import math
import time
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

R = 8.31446261815324
EPS = 1e-12


@dataclass(frozen=True)
class CSSModelConfig:
    # same-platform packed column
    length_m: float = 0.26
    diameter_m: float = 0.021
    adsorbent_mass_kg: float = 0.036
    particle_equivalent_diameter_m: float = 0.0014
    bed_void_fraction: float = 0.38
    # transport / thermal model priors
    gas_viscosity_Pa_s: float = 1.80e-5
    molecular_dispersion_m2_s: float = 1.0e-5
    dispersion_velocity_factor: float = 0.50
    solid_heat_capacity_J_kg_K: float = 900.0
    gas_heat_capacity_J_mol_K: float = 30.0
    effective_axial_conductivity_W_m_K: float = 0.15
    wall_heat_transfer_W_m2_K: float = 8.0
    heat_ads_CO2_J_mol: float = 28000.0
    heat_ads_N2_J_mol: float = 16000.0
    # ZLC-derived transport prior: k_LDF = 15 D/R^2 for a spherical LDF mapping
    k_LDF_CO2_s: float = 15.0 * 4.36e-2
    k_LDF_N2_s: float = 15.0 * 4.36e-2 * math.sqrt(44.01 / 28.014)
    kinetic_multiplier: float = 1.0
    dispersion_multiplier: float = 1.0
    # numerics
    n_cells: int = 10
    rtol: float = 2.0e-5
    atol: float = 1.0e-8
    max_step_s: float = 2.0
    min_css_cycles: int = 8
    max_css_cycles: int = 30
    css_consecutive_cycles: int = 3
    css_kpi_tol_fraction: float = 1.0e-3  # 0.1 percentage point
    css_state_rel_tol: float = 2.0e-3
    pressure_event_rel_tol: float = 5.0e-3
    max_pressure_step_s: float = 1200.0
    ambient_temperature_K: float = 295.15
    ambient_pressure_bar: float = 1.0
    vacuum_efficiency: float = 0.60
    open_boundary_velocity_cap_m_s: float = 0.03
    # numerical fail-safe controls (do not relax scientific validation gates)
    integration_chunk_s: float = 20.0
    step_wall_time_s: float = 60.0
    case_wall_time_s: float = 180.0
    strict_pressure_events: bool = True

    @property
    def area_m2(self) -> float:
        return math.pi * self.diameter_m**2 / 4.0

    @property
    def bed_volume_m3(self) -> float:
        return self.area_m2 * self.length_m

    @property
    def bulk_adsorbent_density_kg_m3(self) -> float:
        return self.adsorbent_mass_kg / self.bed_volume_m3

    @property
    def wall_area_per_bed_volume_m_inv(self) -> float:
        return 4.0 / self.diameter_m


@dataclass(frozen=True)
class CycleOperatingPoint:
    case_id: str
    feed_co2_fraction: float
    high_pressure_bar: float
    intermediate_pressure_bar: float
    low_pressure_bar: float
    adsorption_time_s: float
    feed_superficial_velocity_m_s: float
    timing_scale: float = 1.0
    lpp_feed_co2_fraction_fallback: float = 0.02
    source_role: str = "model_case"


@dataclass
class StepResult:
    name: str
    t_start: float
    t_end: float
    reached_event: bool
    state: np.ndarray
    mean_pressure_bar: float
    nfev: int


class EquilibriumTable:
    """Fast trilinear interpolation of an external binary equilibrium callback.

    callback(y_co2, pressure_bar, temperature_K) -> dict containing q_co2/q_n2
    """
    def __init__(self, callback: Callable[[float,float,float], Dict[str,float]],
                 p_range=(0.008,1.10), t_range=(285.0,315.0),
                 n_y=25, n_p=28, n_t=7):
        from scipy.interpolate import RegularGridInterpolator
        self.ys=np.linspace(1.0e-5,0.99999,n_y)
        self.ps=np.geomspace(max(p_range[0],1.0e-4),max(p_range[1],p_range[0]*1.01),n_p)
        self.ts=np.linspace(t_range[0],t_range[1],n_t)
        qc=np.empty((n_y,n_p,n_t)); qn=np.empty_like(qc)
        for i,y in enumerate(self.ys):
            for j,p in enumerate(self.ps):
                for k,t in enumerate(self.ts):
                    z=callback(float(y),float(p),float(t))
                    qc[i,j,k]=max(float(z.get("q_co2",0.0)),0.0)
                    qn[i,j,k]=max(float(z.get("q_n2",0.0)),0.0)
        self._qc=RegularGridInterpolator((self.ys,self.ps,self.ts),qc,bounds_error=False,fill_value=None)
        self._qn=RegularGridInterpolator((self.ys,self.ps,self.ts),qn,bounds_error=False,fill_value=None)
        self.pmin,self.pmax=float(self.ps[0]),float(self.ps[-1])
        self.tmin,self.tmax=float(self.ts[0]),float(self.ts[-1])
    def evaluate(self,y,p,t):
        y=np.clip(np.asarray(y,float),self.ys[0],self.ys[-1])
        p=np.clip(np.asarray(p,float),self.pmin,self.pmax)
        t=np.clip(np.asarray(t,float),self.tmin,self.tmax)
        pts=np.column_stack([y.ravel(),p.ravel(),t.ravel()])
        return self._qc(pts).reshape(y.shape), self._qn(pts).reshape(y.shape)


class NonIsothermalCSSSolver:
    # state layout per cell: cCO2, cN2, T, qCO2, qN2; then 8 accumulators
    NACC=8
    I_NET_CO2=0; I_NET_N2=1; I_FEED_CO2=2; I_FEED_TOT=3
    I_PROD_CO2=4; I_PROD_TOT=5; I_RAFF_CO2=6; I_RAFF_TOT=7

    def __init__(self, eq: EquilibriumTable, cfg: CSSModelConfig):
        self.eq=eq; self.cfg=cfg
        self.n=int(cfg.n_cells); self.dz=cfg.length_m/self.n
        self.rho_b=cfg.bulk_adsorbent_density_kg_m3
        self.alpha=150.0*cfg.gas_viscosity_Pa_s*(1-cfg.bed_void_fraction)**2/(cfg.bed_void_fraction**3*cfg.particle_equivalent_diameter_m**2)
        self._jac_sparsity=self._build_jac_sparsity()

    def _build_jac_sparsity(self):
        """Block-tridiagonal sparsity for finite-volume BDF Jacobians.

        Each physical cell depends on itself and its immediate neighbours.
        Cycle accumulators depend on boundary-cell fluxes but do not feed back
        into the physical state.  Supplying this pattern makes SciPy's BDF
        finite-difference Jacobian substantially faster and more stable.
        """
        from scipy.sparse import lil_matrix
        nvar=5*self.n+self.NACC; S=lil_matrix((nvar,nvar),dtype=int)
        for i in range(self.n):
            rows=range(5*i,5*i+5)
            for j in range(max(0,i-1),min(self.n,i+2)):
                cols=range(5*j,5*j+5)
                for r in rows:
                    for c in cols: S[r,c]=1
        # accumulator derivatives depend only on the first/last cells
        for r in range(5*self.n,nvar):
            for c in list(range(0,5))+list(range(5*(self.n-1),5*self.n)): S[r,c]=1
        return S.tocsr()

    def _atol_vector(self):
        # Scale-aware absolute tolerances reduce false stiffness across
        # concentration (~10^1), temperature (~10^2), loading (~1), and
        # integrated-mole (~10^-3) states.
        a=np.empty(5*self.n+self.NACC,float)
        for i in range(self.n):
            b=5*i; a[b:b+2]=max(self.cfg.atol,1e-9); a[b+2]=max(self.cfg.atol,1e-6); a[b+3:b+5]=max(self.cfg.atol,1e-9)
        a[5*self.n:]=max(self.cfg.atol,1e-11)
        return a

    def _unpack(self,Y):
        n=self.n; X=Y[:5*n].reshape(n,5); a=Y[5*n:5*n+self.NACC]
        return X,a

    def _stored_component_moles(self,Y):
        X,_=self._unpack(Y); c1=X[:,0]; c2=X[:,1]; q1=X[:,3]; q2=X[:,4]
        Vc=self.cfg.area_m2*self.dz
        return np.array([
            np.sum(self.cfg.bed_void_fraction*c1*Vc + self.rho_b*q1*Vc),
            np.sum(self.cfg.bed_void_fraction*c2*Vc + self.rho_b*q2*Vc),
        ])

    def _mean_pressure_bar(self,Y):
        X,_=self._unpack(Y); c=np.maximum(X[:,0]+X[:,1],1e-9); T=np.clip(X[:,2],250,450)
        return float(np.mean(c*R*T)/1e5)

    def _ergun_velocity(self,grad_drive_Pa_m,ctot,T,y):
        # grad_drive = -dP/dz; positive drives flow in +z
        G=float(grad_drive_Pa_m)
        if abs(G)<1e-10: return 0.0
        M=y*0.04401+(1-y)*0.028014
        rho=max(float(ctot)*float(M),1e-6)
        beta=1.75*rho*(1-self.cfg.bed_void_fraction)/(self.cfg.bed_void_fraction**3*self.cfg.particle_equivalent_diameter_m)
        mag=(-self.alpha+math.sqrt(self.alpha*self.alpha+4*max(beta,EPS)*abs(G)))/(2*max(beta,EPS))
        return math.copysign(max(mag,0.0),G)

    def initial_state(self,op: CycleOperatingPoint, initial_y: Optional[float]=None):
        y=float(op.feed_co2_fraction if initial_y is None else initial_y)
        T=np.full(self.n,self.cfg.ambient_temperature_K)
        P=np.full(self.n,op.high_pressure_bar)
        ctot=P*1e5/(R*T)
        c1=ctot*y; c2=ctot*(1-y)
        q1,q2=self.eq.evaluate(np.full(self.n,y),P,T)
        X=np.column_stack([c1,c2,T,q1,q2])
        return np.r_[X.ravel(),np.zeros(self.NACC)]

    def _boundary_flux(self,u,c1,c2,T,res_y,res_T):
        # convective molar component / enthalpy flux; positive in +z
        if u>=0:
            yy=res_y; tt=res_T; ct=c1+c2 if not np.isfinite(res_y) else None
            if ct is None:
                # reservoir concentration supplied elsewhere by caller
                raise RuntimeError("internal use only")
        return None

    def _rhs_factory(self,step: str,op: CycleOperatingPoint,res_y_right: float=0.02):
        cfg=self.cfg; n=self.n; dz=self.dz; A=cfg.area_m2; eps=cfg.bed_void_fraction; rho_b=self.rho_b
        Tamb=cfg.ambient_temperature_K; cp=cfg.gas_heat_capacity_J_mol_K
        step=step.lower()
        def rhs(t,Y):
            X,acc=self._unpack(Y)
            c1=np.maximum(X[:,0],1e-9); c2=np.maximum(X[:,1],1e-9)
            T=np.clip(X[:,2],250.0,450.0); q1=np.maximum(X[:,3],0.0); q2=np.maximum(X[:,4],0.0)
            ct=np.maximum(c1+c2,1e-9); y=np.clip(c1/ct,1e-8,1-1e-8); P=ct*R*T
            pbar=P/1e5
            q1e,q2e=self.eq.evaluate(y,pbar,T)
            k1=cfg.k_LDF_CO2_s*cfg.kinetic_multiplier; k2=cfg.k_LDF_N2_s*cfg.kinetic_multiplier
            dq1=k1*(q1e-q1); dq2=k2*(q2e-q2)
            # face velocities and component/energy fluxes
            u=np.zeros(n+1); F1=np.zeros(n+1); F2=np.zeros(n+1); FE=np.zeros(n+1)
            # During the pressure-controlled adsorption step the inlet superficial
            # velocity is prescribed experimentally, but the outlet molar flow need
            # not equal the inlet flow while the solid is taking up gas.  A uniform
            # velocity at every face artificially convects too much gas through the
            # bed and can depress cyclic recovery.  Use a quasi-isobaric sink-balanced
            # total-molar-flux profile: the inlet flux is fixed and each downstream
            # face is reduced by the local solid molar uptake.  This is a standard
            # finite-volume closure for a pressure-controlled adsorption step and keeps
            # the full transient component balances/mass-closure audit active.
            ads_Ftot=None
            if step=="adsorption":
                ct_feed=(op.high_pressure_bar*1e5)/(R*Tamb)
                ads_Ftot=np.zeros(n+1,float); ads_Ftot[0]=max(op.feed_superficial_velocity_m_s,0.0)*ct_feed
                for ii in range(n):
                    uptake_areal=rho_b*(dq1[ii]+dq2[ii])*dz
                    ads_Ftot[ii+1]=max(ads_Ftot[ii]-uptake_areal,0.0)
            # internal faces
            for f in range(1,n):
                Pl,Pr=P[f-1],P[f]; ctl,ctr=ct[f-1],ct[f]; Tl,Tr=T[f-1],T[f]; yl,yr=y[f-1],y[f]
                if step=="adsorption":
                    uf=float(ads_Ftot[f]/max(ctl,1e-9))
                else:
                    uf=self._ergun_velocity((Pl-Pr)/dz,0.5*(ctl+ctr),0.5*(Tl+Tr),0.5*(yl+yr))
                    uf=float(np.clip(uf,-cfg.open_boundary_velocity_cap_m_s,cfg.open_boundary_velocity_cap_m_s))
                u[f]=uf
                if uf>=0: cup1,cup2,Tup= c1[f-1],c2[f-1],Tl
                else: cup1,cup2,Tup= c1[f],c2[f],Tr
                D=cfg.dispersion_multiplier*(cfg.molecular_dispersion_m2_s+cfg.dispersion_velocity_factor*cfg.particle_equivalent_diameter_m*abs(uf))
                F1[f]=uf*cup1-D*(c1[f]-c1[f-1])/dz
                F2[f]=uf*cup2-D*(c2[f]-c2[f-1])/dz
                FE[f]=uf*(cup1+cup2)*cp*Tup-cfg.effective_axial_conductivity_W_m_K*(Tr-Tl)/dz
            # boundaries: define reservoir pressures/compositions
            PH=op.high_pressure_bar*1e5; PI=op.intermediate_pressure_bar*1e5; PL=op.low_pressure_bar*1e5
            # left face
            if step=="adsorption":
                u[0]=max(op.feed_superficial_velocity_m_s,0.0)
                # High pressure is pressure-controlled experimentally.  With the
                # adsorption step represented at prescribed superficial velocity,
                # use the measured PH reservoir concentration directly rather than
                # adding an Ergun half-cell pressure increment (which would double
                # count pressure drop and grossly inflate the feed inventory).
                Pres=PH
                yres=op.feed_co2_fraction; Tres=Tamb; ctres=Pres/(R*Tres)
            elif step=="evacuation":
                Pres=PL; yres=op.feed_co2_fraction; Tres=Tamb; ctres=Pres/(R*Tres)
                u[0]=self._ergun_velocity((Pres-P[0])/(dz/2),0.5*(ctres+ct[0]),0.5*(Tres+T[0]),0.5*(yres+y[0])); u[0]=float(np.clip(u[0],-cfg.open_boundary_velocity_cap_m_s,cfg.open_boundary_velocity_cap_m_s))
            else:
                Pres=P[0]; yres=y[0]; Tres=T[0]; ctres=ct[0]; u[0]=0.0
            if u[0]>=0:
                F1[0]=u[0]*ctres*yres; F2[0]=u[0]*ctres*(1-yres); FE[0]=u[0]*ctres*cp*Tres
            else:
                F1[0]=u[0]*c1[0]; F2[0]=u[0]*c2[0]; FE[0]=u[0]*ct[0]*cp*T[0]
            # right face
            if step in {"adsorption","blowdown","lpp"}:
                if step=="adsorption": Pres=PH; yres=max(min(res_y_right,0.2),1e-5)
                elif step=="blowdown": Pres=PI; yres=max(min(res_y_right,0.2),1e-5)
                else: Pres=PH; yres=max(min(res_y_right,0.2),1e-5)
                Tres=Tamb; ctres=Pres/(R*Tres)
                if step=="adsorption":
                    u[n]=float(ads_Ftot[n]/max(ct[-1],1e-9))
                else:
                    u[n]=self._ergun_velocity((P[-1]-Pres)/(dz/2),0.5*(ct[-1]+ctres),0.5*(T[-1]+Tres),0.5*(y[-1]+yres)); u[n]=float(np.clip(u[n],-cfg.open_boundary_velocity_cap_m_s,cfg.open_boundary_velocity_cap_m_s))
            else:
                Pres=P[-1]; yres=y[-1]; Tres=T[-1]; ctres=ct[-1]; u[n]=0.0
            if u[n]>=0:
                F1[n]=u[n]*c1[-1]; F2[n]=u[n]*c2[-1]; FE[n]=u[n]*ct[-1]*cp*T[-1]
            else:
                F1[n]=u[n]*ctres*yres; F2[n]=u[n]*ctres*(1-yres); FE[n]=u[n]*ctres*cp*Tres
            dc1=-(F1[1:]-F1[:-1])/(eps*dz)-(rho_b/eps)*dq1
            dc2=-(F2[1:]-F2[:-1])/(eps*dz)-(rho_b/eps)*dq2
            Cvol=np.maximum(eps*ct*cp+rho_b*cfg.solid_heat_capacity_J_kg_K,1e4)
            qheat=rho_b*(cfg.heat_ads_CO2_J_mol*dq1+cfg.heat_ads_N2_J_mol*dq2)
            hvol=cfg.wall_heat_transfer_W_m2_K*cfg.wall_area_per_bed_volume_m_inv
            dT=(-(FE[1:]-FE[:-1])/dz+qheat-hvol*(T-Tamb))/Cvol
            dX=np.column_stack([dc1,dc2,dT,dq1,dq2])
            # signed external molar input to domain
            net1=A*(F1[0]-F1[-1]); net2=A*(F2[0]-F2[-1])
            feed1=A*max(F1[0],0.0) if step=="adsorption" else 0.0
            feedt=A*max(F1[0]+F2[0],0.0) if step=="adsorption" else 0.0
            prod1=A*max(-F1[0],0.0) if step=="evacuation" else 0.0
            prodt=A*max(-(F1[0]+F2[0]),0.0) if step=="evacuation" else 0.0
            raff1=A*max(F1[-1],0.0) if step=="adsorption" else 0.0
            rafft=A*max(F1[-1]+F2[-1],0.0) if step=="adsorption" else 0.0
            dacc=np.array([net1,net2,feed1,feedt,prod1,prodt,raff1,rafft])
            return np.r_[dX.ravel(),dacc]
        return rhs

    def _sanitize_state(self, Z):
        """Project only numerically invalid physical states back to admissible bounds.

        Any projection-induced inventory change is still exposed by the cycle-level
        component mass-closure audit, so this is a numerical fail-safe rather than
        a hidden scientific correction.
        """
        Z=np.asarray(Z,float).copy()
        X,a=self._unpack(Z)
        X[:,0:2]=np.maximum(X[:,0:2],1e-10)
        X[:,2]=np.clip(X[:,2],250.0,450.0)
        X[:,3:5]=np.maximum(X[:,3:5],0.0)
        return np.r_[X.ravel(),a]

    def _integrate_segment(self, Y, ta, tb, rhs, event=None, deadline=None):
        """Integrate one short segment with a conservative stiff-solver ladder.

        BDF remains the primary method. Radau is used only when BDF fails with
        step-size/stiffness pathologies. A shared wall-clock deadline prevents a
        single pathological segment from blocking the complete publication run.
        """
        attempts=[('BDF',1.0),('BDF',0.25),('Radau',0.50)]
        errors=[]
        for method,fac in attempts:
            if deadline is not None and time.perf_counter()>=deadline:
                raise RuntimeError('CSS integration wall-time guard reached before a stable segment solution was obtained')
            def guarded_rhs(t,z):
                if deadline is not None and time.perf_counter()>=deadline:
                    raise RuntimeError('CSS integration wall-time guard reached during RHS evaluation')
                out=rhs(t,z)
                if not np.all(np.isfinite(out)):
                    raise FloatingPointError('non-finite derivative encountered in CSS RHS')
                return out
            kwargs=dict(method=method,rtol=self.cfg.rtol,atol=self._atol_vector(),
                        max_step=max(min(self.cfg.max_step_s*fac,tb-ta),1e-4))
            if method in {'BDF','Radau'}:
                kwargs['jac_sparsity']=self._jac_sparsity
            if event is not None:
                kwargs['events']=event
            try:
                sol=solve_ivp(guarded_rhs,(ta,tb),self._sanitize_state(Y),**kwargs)
                if sol.success and len(sol.t):
                    return sol
                errors.append(f'{method}(x{fac:g}): {getattr(sol,"message","unsuccessful integration")}')
            except Exception as exc:
                errors.append(f'{method}(x{fac:g}): {exc!r}')
                if 'wall-time guard' in repr(exc):
                    raise RuntimeError(str(exc)) from exc
        raise RuntimeError('CSS segment integration failed after BDF/Radau retry ladder: '+' | '.join(errors))

    def _integrate_step(self,Y,t0,step,op,res_y_right,max_duration=None):
        rhs=self._rhs_factory(step,op,res_y_right)
        step_deadline=time.perf_counter()+max(float(self.cfg.step_wall_time_s),1.0)
        case_deadline=getattr(self,'_case_deadline',None)
        deadline=min(step_deadline,case_deadline) if case_deadline is not None else step_deadline
        reached=False
        if step=='adsorption':
            total=max(op.adsorption_time_s*op.timing_scale,1e-3)
            target_time=t0+total
            Z=self._sanitize_state(Y); ta=float(t0); nfev=0
            chunk=max(min(float(self.cfg.integration_chunk_s),total),1e-3)
            while ta < target_time-1e-10:
                tb=min(target_time,ta+chunk)
                sol=self._integrate_segment(Z,ta,tb,rhs,event=None,deadline=deadline)
                Z=self._sanitize_state(sol.y[:,-1]); ta=float(sol.t[-1]); nfev+=int(sol.nfev)
            return StepResult(step,t0,ta,False,Z,self._mean_pressure_bar(Z),nfev)

        target={'blowdown':op.intermediate_pressure_bar*(1+self.cfg.pressure_event_rel_tol),
                'evacuation':op.low_pressure_bar*(1+self.cfg.pressure_event_rel_tol),
                'lpp':op.high_pressure_bar*(1-self.cfg.pressure_event_rel_tol)}[step]
        direction=-1 if step in {'blowdown','evacuation'} else 1
        def ev(t,z): return self._mean_pressure_bar(z)-target
        ev.terminal=True; ev.direction=direction
        total=float(max_duration or self.cfg.max_pressure_step_s)
        target_time=t0+total; Z=self._sanitize_state(Y); ta=float(t0); nfev=0
        chunk=max(min(float(self.cfg.integration_chunk_s),total),1e-3)
        last_p=self._mean_pressure_bar(Z); stagnant=0
        while ta < target_time-1e-10:
            tb=min(target_time,ta+chunk)
            sol=self._integrate_segment(Z,ta,tb,rhs,event=ev,deadline=deadline)
            Z=self._sanitize_state(sol.y[:,-1]); ta=float(sol.t[-1]); nfev+=int(sol.nfev)
            if len(sol.t_events) and len(sol.t_events[0]):
                reached=True; break
            pnow=self._mean_pressure_bar(Z)
            # Accept an event missed by root-finding only when the terminal state is
            # already within the predeclared pressure tolerance.
            if abs(pnow-target) <= max(abs(target)*self.cfg.pressure_event_rel_tol,5e-5):
                reached=True; break
            progress=(last_p-pnow)*direction*-1.0 if direction<0 else (pnow-last_p)
            if progress <= max(abs(last_p)*1e-7,1e-8): stagnant += 1
            else: stagnant = 0
            last_p=pnow
            if stagnant>=4:
                break
        return StepResult(step,t0,ta,reached,Z,self._mean_pressure_bar(Z),nfev)

    def simulate_cycle(self,Y0,op: CycleOperatingPoint):
        # reset cycle accumulators, keep physical state
        Y=Y0.copy(); Y[-self.NACC:]=0.0
        stored0=self._stored_component_moles(Y)
        t=0.0; steps=[]
        # adsorption
        s=self._integrate_step(Y,t,"adsorption",op,op.lpp_feed_co2_fraction_fallback); steps.append(s); Y=s.state; t=s.t_end
        _,a=self._unpack(Y); raff_y=float(a[self.I_RAFF_CO2]/max(a[self.I_RAFF_TOT],EPS)) if a[self.I_RAFF_TOT]>EPS else op.lpp_feed_co2_fraction_fallback
        # LPP uses the N2-rich light product from the companion column.  The integrated
        # adsorption-outlet composition can be biased upward by late breakthrough; cap it
        # by the explicit light-product prior rather than recycling a CO2-rich average.
        raff_y=float(np.clip(min(raff_y,op.lpp_feed_co2_fraction_fallback),1e-4,0.20))
        for name in ["blowdown","evacuation","lpp"]:
            s=self._integrate_step(Y,t,name,op,raff_y); steps.append(s); Y=s.state; t=s.t_end
            if self.cfg.strict_pressure_events and not s.reached_event:
                raise RuntimeError(f"CSS pressure step {name} did not reach its target within numerical/physical guards; final mean pressure={s.final_mean_pressure_bar:.6g} bar")
        stored1=self._stored_component_moles(Y); _,a=self._unpack(Y)
        prod1=max(float(a[self.I_PROD_CO2]),0.0); prodt=max(float(a[self.I_PROD_TOT]),EPS); feed1=max(float(a[self.I_FEED_CO2]),EPS)
        purity=prod1/prodt; recovery=min(prod1/feed1,1.5)
        # vacuum work estimate from evacuated heavy product at PL; adds co-current blowdown conservatively through final product moles only.
        # This is an integrated ideal-gas/isothermal thermodynamic estimate, not a measured pump curve.
        eta=max(self.cfg.vacuum_efficiency,1e-3)
        Tref=self.cfg.ambient_temperature_K; Patm=self.cfg.ambient_pressure_bar
        work_J=prodt*R*Tref*max(math.log(Patm/max(op.low_pressure_bar,1e-4)),0.0)/eta
        prod_m3_s=prod1/max(self.cfg.bed_volume_m3*t,EPS)
        net=np.array([a[self.I_NET_CO2],a[self.I_NET_N2]],float)
        residual=stored1-stored0-net
        denom=np.maximum(np.abs(stored0)+np.abs(net),1e-9)
        closure=float(np.max(np.abs(residual)/denom))
        metrics={
            "purity_fraction":purity,"recovery_fraction":recovery,
            "productivity_mol_m3_s":prod_m3_s,
            "cycle_time_s":t,"product_co2_mol":prod1,"feed_co2_mol":feed1,
            "vacuum_work_kJ_mol_CO2":work_J/max(prod1,EPS)/1000.0,
            "component_mass_closure_rel_max":closure,
            "lpp_feed_co2_fraction":raff_y,
            "temperature_min_K":float(np.min(self._unpack(Y)[0][:,2])),
            "temperature_max_K":float(np.max(self._unpack(Y)[0][:,2])),
            "final_mean_pressure_bar":self._mean_pressure_bar(Y),
            "all_pressure_events_reached":bool(all(s.reached_event for s in steps[1:])),
            "nfev_cycle":int(sum(s.nfev for s in steps)),
        }
        return Y,metrics,steps

    def run_to_css(self,op: CycleOperatingPoint,Y0=None):
        Y=self.initial_state(op) if Y0 is None else np.asarray(Y0,float).copy()
        rows=[]; consecutive=0; prev=None; prev_phys=None
        css=False
        self._case_deadline=time.perf_counter()+max(float(self.cfg.case_wall_time_s),1.0)
        for cyc in range(1,self.cfg.max_css_cycles+1):
            if time.perf_counter()>=self._case_deadline:
                raise RuntimeError(f"CSS case {op.case_id} exceeded the case wall-time guard before reaching CSS")
            Ynew,m,steps=self.simulate_cycle(Y,op)
            phys=Ynew[:-self.NACC].copy()
            state_rel=float(np.linalg.norm(phys-(prev_phys if prev_phys is not None else phys))/max(np.linalg.norm(phys),1e-9)) if prev_phys is not None else np.nan
            if prev is not None:
                dp=abs(m["purity_fraction"]-prev["purity_fraction"]); dr=abs(m["recovery_fraction"]-prev["recovery_fraction"])
            else: dp=dr=np.nan
            row={"cycle":cyc,**m,"delta_purity_fraction":dp,"delta_recovery_fraction":dr,"state_relative_change":state_rel}
            rows.append(row)
            if cyc>=self.cfg.min_css_cycles and np.isfinite(dp) and np.isfinite(dr) and dp<=self.cfg.css_kpi_tol_fraction and dr<=self.cfg.css_kpi_tol_fraction and state_rel<=self.cfg.css_state_rel_tol and m["component_mass_closure_rel_max"]<=0.005 and m["all_pressure_events_reached"]:
                consecutive+=1
            else: consecutive=0
            if consecutive>=self.cfg.css_consecutive_cycles:
                css=True; Y=Ynew; break
            prev=m; prev_phys=phys; Y=Ynew
        hist=pd.DataFrame(rows)
        final=rows[-1].copy(); final.update({"css_converged":css,"css_cycles":len(rows),"case_id":op.case_id,**asdict(op)})
        self._case_deadline=None
        return {"final_state":Y,"history":hist,"summary":final}


def validation_score(sim: Dict[str,Any], exp: Dict[str,float]) -> Dict[str,float]:
    s=sim["summary"]
    out={}
    for key,sk in [("purity_fraction","purity_fraction"),("recovery_fraction","recovery_fraction"),("productivity_mol_m3_s","productivity_mol_m3_s")]:
        ev=float(exp.get(key,np.nan)); sv=float(s.get(sk,np.nan))
        out[f"sim_{key}"]=sv; out[f"exp_{key}"]=ev; out[f"abs_error_{key}"]=abs(sv-ev) if np.isfinite(ev) else np.nan
        out[f"relative_error_{key}"]=abs(sv-ev)/max(abs(ev),EPS) if np.isfinite(ev) else np.nan
    return out


def run_validation_suite(eq: EquilibriumTable, base_cfg: CSSModelConfig, validation_cases: pd.DataFrame,
                         mesh_cells: Sequence[int]=(6,10,14), tolerance_factors: Sequence[float]=(2.0,1.0,0.5), logger=None):
    """Validate against published dry same-rig cycles and perform mesh/tolerance QA."""
    case_rows=[]; histories={}
    for _,r in validation_cases.iterrows():
        op=CycleOperatingPoint(
            case_id=str(r["case_id"]),feed_co2_fraction=float(r.get("feed_co2_fraction",0.15)),
            high_pressure_bar=float(r["high_pressure_bar"]),intermediate_pressure_bar=float(r["intermediate_pressure_bar"]),
            low_pressure_bar=float(r["low_pressure_bar"]),adsorption_time_s=float(r["adsorption_time_s"]),
            feed_superficial_velocity_m_s=float(r["feed_superficial_velocity_m_s"]),source_role=str(r.get("evidence_role","published_validation")))
        cfg=base_cfg
        if logger: logger.info("CSS validation case %s starting",op.case_id)
        tcase=time.perf_counter()
        try:
            sol=NonIsothermalCSSSolver(eq,cfg).run_to_css(op)
            score=validation_score(sol,{"purity_fraction":float(r["purity_fraction"]),"recovery_fraction":float(r["recovery_fraction"]),"productivity_mol_m3_s":float(r["productivity_mol_m3_s"])})
            case_rows.append({**sol["summary"],**score,"evidence_role":r.get("evidence_role",""),"source_id":r.get("source_id",""),"solver_status":"OK","solver_error":""})
            histories[op.case_id]=sol["history"]
            if logger: logger.info("CSS validation case %s completed in %.1f s (converged=%s)",op.case_id,time.perf_counter()-tcase,sol["summary"].get("css_converged"))
        except Exception as exc:
            if logger: logger.exception("CSS validation case %s failed after %.1f s",op.case_id,time.perf_counter()-tcase)
            case_rows.append({"case_id":op.case_id,"css_converged":False,"component_mass_closure_rel_max":np.nan,"purity_fraction":np.nan,"recovery_fraction":np.nan,"productivity_mol_m3_s":np.nan,"abs_error_purity_fraction":np.nan,"abs_error_recovery_fraction":np.nan,"evidence_role":r.get("evidence_role",""),"source_id":r.get("source_id",""),"solver_status":"FAIL","solver_error":repr(exc)})
    # QA around first validation case only (sufficient to verify discretization before portfolio).
    # If a required validation case already failed numerically, discretization QA cannot
    # rescue the gate and is genuinely skipped to avoid wasting hours on an invalid state.
    qa=[]
    prelim=pd.DataFrame(case_rows)
    required_solver_ok=(len(prelim)==len(validation_cases) and len(prelim)>0 and
                        prelim.get("solver_status",pd.Series(dtype=str)).astype(str).eq("OK").all() and
                        prelim.get("css_converged",pd.Series(dtype=bool)).fillna(False).all())
    if len(validation_cases) and required_solver_ok:
        r=validation_cases.iloc[0]
        op=CycleOperatingPoint(str(r["case_id"]),float(r.get("feed_co2_fraction",0.15)),float(r["high_pressure_bar"]),float(r["intermediate_pressure_bar"]),float(r["low_pressure_bar"]),float(r["adsorption_time_s"]),float(r["feed_superficial_velocity_m_s"]))
        for nc in mesh_cells:
            cfg=replace(base_cfg,n_cells=int(nc))
            if logger: logger.info("CSS mesh QA starting: n_cells=%s",nc)
            try:
                sol=NonIsothermalCSSSolver(eq,cfg).run_to_css(op); qa.append({"qa_type":"mesh","value":nc,**{k:sol["summary"].get(k,np.nan) for k in ["purity_fraction","recovery_fraction","productivity_mol_m3_s","component_mass_closure_rel_max","css_converged","css_cycles"]},"solver_status":"OK","solver_error":""})
                if logger: logger.info("CSS mesh QA completed: n_cells=%s converged=%s",nc,sol["summary"].get("css_converged"))
            except Exception as exc:
                qa.append({"qa_type":"mesh","value":nc,"css_converged":False,"solver_status":"FAIL","solver_error":repr(exc)})
                if logger: logger.exception("CSS mesh QA failed: n_cells=%s",nc)
        for f in tolerance_factors:
            cfg=replace(base_cfg,rtol=base_cfg.rtol*float(f),atol=base_cfg.atol*float(f))
            if logger: logger.info("CSS tolerance QA starting: factor=%s",f)
            try:
                sol=NonIsothermalCSSSolver(eq,cfg).run_to_css(op); qa.append({"qa_type":"solver_tolerance","value":f,**{k:sol["summary"].get(k,np.nan) for k in ["purity_fraction","recovery_fraction","productivity_mol_m3_s","component_mass_closure_rel_max","css_converged","css_cycles"]},"solver_status":"OK","solver_error":""})
                if logger: logger.info("CSS tolerance QA completed: factor=%s converged=%s",f,sol["summary"].get("css_converged"))
            except Exception as exc:
                qa.append({"qa_type":"solver_tolerance","value":f,"css_converged":False,"solver_status":"FAIL","solver_error":repr(exc)})
                if logger: logger.exception("CSS tolerance QA failed: factor=%s",f)
    elif len(validation_cases):
        qa.append({"qa_type":"skipped","value":np.nan,"css_converged":False,"solver_status":"SKIPPED",
                   "solver_error":"mesh/tolerance QA skipped because at least one required same-rig validation case failed or did not reach CSS"})
        if logger: logger.warning("CSS mesh/tolerance QA skipped because required same-rig validation did not complete and converge")
    return pd.DataFrame(case_rows),pd.DataFrame(qa),histories


def assess_validation_gate(summary: pd.DataFrame, qa: pd.DataFrame) -> Dict[str,Any]:
    """Predeclared publication-closure gate; failure is reported, never hidden."""
    if summary is None or summary.empty:
        return {"status":"FAIL","reason":"no validation simulations"}
    css_col=summary["css_converged"] if "css_converged" in summary.columns else pd.Series(False,index=summary.index)
    mass_col=summary["component_mass_closure_rel_max"] if "component_mass_closure_rel_max" in summary.columns else pd.Series(np.nan,index=summary.index)
    solver_col=summary["solver_status"] if "solver_status" in summary.columns else pd.Series("OK",index=summary.index)
    solver_ok=bool(solver_col.astype(str).eq("OK").all())
    css_ok=bool(css_col.fillna(False).all()) and solver_ok
    mass_vals=pd.to_numeric(mass_col,errors="coerce")
    mass_ok=bool(mass_vals.notna().all() and (mass_vals<=0.005).all())
    # Experimental KPI acceptance intentionally moderate because the model is not fitted to these validation KPIs.
    pur_err=pd.to_numeric(summary.get("abs_error_purity_fraction",np.nan),errors="coerce")
    rec_err=pd.to_numeric(summary.get("abs_error_recovery_fraction",np.nan),errors="coerce")
    kpi_ok=bool((pur_err<=0.05).all() and (rec_err<=0.08).all())
    mesh_ok=True; tol_ok=True
    if qa is not None and len(qa):
        for typ,limit in [("mesh",0.002),("solver_tolerance",0.002)]:
            q=qa[qa["qa_type"]==typ] if "qa_type" in qa.columns else pd.DataFrame()
            if len(q)>=2:
                if "solver_status" in q.columns and not q["solver_status"].astype(str).eq("OK").all():
                    if typ=="mesh": mesh_ok=False
                    else: tol_ok=False
                for col in ["purity_fraction","recovery_fraction"]:
                    vals=pd.to_numeric(q[col],errors="coerce").dropna() if col in q.columns else pd.Series(dtype=float)
                    if len(vals)<2 or float(vals.max()-vals.min())>limit:
                        if typ=="mesh": mesh_ok=False
                        else: tol_ok=False
    status="PASS" if all([solver_ok,css_ok,mass_ok,kpi_ok,mesh_ok,tol_ok]) else "FAIL"
    return {"status":status,"solver_completed_all":solver_ok,"css_converged_all":css_ok,"mass_closure_all_le_0p5pct":mass_ok,"experimental_kpi_gate":kpi_ok,"mesh_gate_0p2pp":mesh_ok,"tolerance_gate_0p2pp":tol_ok,"note":"PASS authorizes the solver as cross-fidelity falsification evidence; FAIL leaves all high-fidelity claims explicitly open."}
