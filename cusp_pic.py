import time
import numpy as np
import matplotlib.pyplot as plt
from multigrid_poisson import MultigridPoisson

# Christian Zuniga, March 2021. Based on Master's thesis "Particle In Cell Simulation
# of Cusp-Confined Plasma With Monte Carlo Collisions".
# Substituted some functions with Python functions from Numpy and Scipy
# 2026: completed electrostatic PIC loop (CIC weighting, multigrid Poisson solve,
# Boris push with E and B, wall boundaries, diagnostics). Collisions not yet added.

# define constants
pi = np.pi
u0 = 4*pi*10**(-7)  # magnetic permeability free space (henrys/ m)
etome = 1.7588196*10**11  # electron |q|/m  TODO use modified charge distribution
md = 1591.05350		#e*e/2 me e0
miontoelec = 256    # artificial ion/electron mass ratio
qe = 1.602176634e-19     # elementary charge (C)
me = qe/etome            # electron mass consistent with etome (kg)
eps0 = 8.8541878128e-12  # permittivity of free space (F/m)

cr = 461341     # check


class PlasmaChamber():
    def __init__(self):
        return None


class Cusp(PlasmaChamber):
    """
    Rectangular chamber [0,Lx] x [0,Ly] (2D, infinite in z) with grounded
    conducting walls. Two wires along z at the bottom corners, (0,0) carrying
    +I and (Lx,0) carrying -I, create the cusp magnetic field.
    Grid: (Nx+1) x (Ny+1) nodes, arrays indexed [ix, iy].
    """
    def __init__(self, I=600, Lx=0.01, Ly=0.01, Nx=128, Ny=128):
        self.LX = Lx		# Lx, Ly are the size of rectangle for simulation (m)
        self.LY = Ly
        self.I = I      # Amperes current in wires
        self.bconst1 = u0*self.I/(2*pi)
        self.Nx = Nx	# number of grid cells (Nx+1 nodes); 2^k cells is fastest for multigrid
        self.Ny = Ny
        self.dx = Lx/self.Nx
        self.dy = Ly/self.Ny
        if not np.isclose(self.dx, self.dy):
            raise ValueError("Poisson solver assumes dx == dy (choose Lx/Nx == Ly/Ny)")
        self.x = np.linspace(0, Lx, Nx+1)
        self.y = np.linspace(0, Ly, Ny+1)
        self.rmin2 = (0.5*self.dx)**2       # avoid the 1/r singularity at the wires

        # Dirichlet points: grounded walls (phi = 0). Electrodes could be added here.
        fixed = np.zeros((Nx+1, Ny+1), dtype=bool)
        fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
        self.wallPotential = np.zeros((Nx+1, Ny+1))
        self.poisson = MultigridPoisson(fixed, self.wallPotential, nu1=2, nu2=2)
        self.phi = None
        self.Ex = np.zeros((Nx+1, Ny+1))
        self.Ey = np.zeros((Nx+1, Ny+1))
        self.mgCycles = 0

    def getMagneticFieldAtPosition(self,x,y):   # works for scalars or arrays (m)
        r = np.maximum(x**2 + y**2, self.rmin2)
        r2 = np.maximum((x-self.LX)**2 + y**2, self.rmin2)
        Bx = self.bconst1*(-y/r + y/r2)      # Magnetic field of 2 wires in z, with current +z for wire at 0, -z wire at Lx
        By = self.bconst1*(x/r + (self.LX-x)/r2)
        return (Bx,By)

    def getMagneticRotMatrixAtPosition(self,P,qtomdt): # receive position matrices and charge/mass ratio x dt/2
        # Legacy rotation-matrix form of the magnetic push; runSimulation uses
        # the equivalent Boris cross-product form instead.
        numCols = 3*P.shape[1]
        Htot = np.zeros((3,numCols))
        Bx,By = self.getMagneticFieldAtPosition(P[0,:],P[1,:])
        a = qtomdt*By
        b = qtomdt*Bx
        a2 = a**2
        b2 = b**2
        onesM = np.ones(a.shape)
        S = a2 + b2  + onesM
        H11 = (b2 - a2 + onesM)/S
        H12 = 2*a*b/S
        H13 = -2*a/S
        H21 = H12
        H22 =  (onesM + a2 - b2)/S
        H23 = 2 *b/S
        H31 = -H13
        H32 = -H23
        H33 = (onesM -a2 - b2)/S
        Htot[0,0:numCols:3] = H11
        Htot[0,1:numCols:3] = H12
        Htot[0,2:numCols:3] = H13
        Htot[1,0:numCols:3] = H21
        Htot[1,1:numCols:3] = H22
        Htot[1,2:numCols:3] = H23
        Htot[2,0:numCols:3] = H31
        Htot[2,1:numCols:3] = H32
        Htot[2,2:numCols:3] = H33
        return Htot # return matrix of  H matrices, size 3, 3Np

    def calculateElectricField(self,chargeDistribution, tol=1e-5):
        """
        Solve  lap(phi) = -rho/eps0  with phi = wall potential on the walls,
        then E = -grad(phi). chargeDistribution is rho on the nodes (C/m^3).
        After the first call the previous phi is used as the initial guess,
        so usually only a couple of V-cycles are needed per time step.
        """
        f = -chargeDistribution*self.dx**2/eps0
        self.phi, info = self.poisson.solve(f, u0=self.phi, tol=tol)
        self.mgCycles = info["cycles"]
        gx, gy = np.gradient(self.phi, self.dx, self.dy)   # central inside, one-sided at walls
        self.Ex = -gx
        self.Ey = -gy
        return self.phi, self.Ex, self.Ey

    def cicWeights(self, pos):
        """Cloud-in-cell (bilinear) node indices and weights for positions (Np,2) in m."""
        gx = pos[:, 0]/self.dx
        gy = pos[:, 1]/self.dy
        i = np.clip(np.floor(gx).astype(int), 0, self.Nx-1)
        j = np.clip(np.floor(gy).astype(int), 0, self.Ny-1)
        fx = gx - i
        fy = gy - j
        ny = self.Ny + 1
        idx = np.stack([i*ny + j, (i+1)*ny + j, i*ny + j + 1, (i+1)*ny + j + 1], axis=1)
        w = np.stack([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy], axis=1)
        return idx, w

    def gatherField(self, idx, w):
        """Interpolate (Ex, Ey) from nodes to particles with the CIC weights."""
        Ex = np.sum(self.Ex.ravel()[idx]*w, axis=1)
        Ey = np.sum(self.Ey.ravel()[idx]*w, axis=1)
        return Ex, Ey

    def checkBoundaries(self, plasma):
        """
        Bottom wall (y < 0, where the wires are): particle is lost and replaced
        by a new one at a random position in the loading region.
        Side walls and top: specular reflection.
        Returns boolean masks (electrons, ions) of particles replaced this step.
        """
        replaced = []
        for pos, vel, vth in ((plasma.loceMatrix, plasma.veleMatrix, plasma.vthermale),
                              (plasma.lociMatrix, plasma.veliMatrix, plasma.vthermali)):
            # reflect at x = 0 and x = Lx
            m = pos[:, 0] < 0
            pos[m, 0] = -pos[m, 0]; vel[m, 0] = -vel[m, 0]
            m = pos[:, 0] > self.LX
            pos[m, 0] = 2*self.LX - pos[m, 0]; vel[m, 0] = -vel[m, 0]
            # reflect at the top
            m = pos[:, 1] > self.LY
            pos[m, 1] = 2*self.LY - pos[m, 1]; vel[m, 1] = -vel[m, 1]
            # lost at the bottom: replace
            lost = pos[:, 1] < 0
            n = int(lost.sum())
            if n:
                pos[lost] = plasma.samplePositions(n, self.LX, self.LY)
                vel[lost] = plasma.sampleVelocities(n, vth)
            # guard against a particle crossing more than one wall in one step
            np.clip(pos[:, 0], 0.0, self.LX, out=pos[:, 0])
            np.clip(pos[:, 1], 0.0, self.LY, out=pos[:, 1])
            replaced.append(lost)
        return replaced[0], replaced[1]


class PICSimulation():
    def __init__(self, T=30*(10**-9), Np=1000, dt=1e-11, n0=1e14, Nx=128, Ny=128,
                 seed=0, fieldTol=1e-5):
        self.Np = Np    # number of particles Np electrons, Np ions
        self.simTime = T   # simulation time (sec)
        self.dt = dt       # timestep (s); must resolve electron gyration and plasma oscillation
        self.Nt = int(round(self.simTime/self.dt))
        print(" Number of time steps ", self.Nt, " time step (s) ", self.dt, " time ", self.simTime)
        self.I = 600   # current for B field (seems excessive)
        self.Lx = 0.01  # width (m)
        self.Ly = 0.01  # height simulation area (m)
        self.n0 = n0    # initial plasma density in the loading region (m^-3)
        self.fieldTol = fieldTol
        self.qme = -etome               # electron q/m
        self.qmi = etome/miontoelec     # ion q/m
        self.qtomdte = self.qme*self.dt/2
        self.qtomdti = self.qmi*self.dt/2
        self.cusp = Cusp(self.I,self.Lx,self.Ly,Nx,Ny)   # init chamber
        self.plasma = Plasma(self.Np,self.Lx,self.Ly,self.n0,seed=seed) # init plasma
        self.trajectory = np.full((self.Nt+1,2), np.nan) # sample electron trajectory (m)
        self.trajectoryi = np.full((self.Nt+1,2), np.nan) # sample ion trajectory (m)
        self.trajectory[0,:] = self.plasma.loceMatrix[0,:]
        self.trajectoryi[0,:] = self.plasma.lociMatrix[0,:]
        self.printStabilityChecks()

        # initial field, then push velocities back half a step (leapfrog: v at t-dt/2)
        self.updateField()
        Ee, Ei, Be, Bi = self.fieldsAtParticles()
        self.plasma.veleMatrix = self.borisPush(self.plasma.veleMatrix, Ee, Be, self.qme, -self.dt/2)
        self.plasma.veliMatrix = self.borisPush(self.plasma.veliMatrix, Ei, Bi, self.qmi, -self.dt/2)

        # diagnostics
        self.time = np.arange(self.Nt)*self.dt
        self.vesq = np.zeros(self.Nt)  # |v|^2 of sample electron, proxy for its kinetic energy
        self.kineticE = np.zeros(self.Nt)  # electron kinetic energy (J/m)
        self.kineticI = np.zeros(self.Nt)  # ion kinetic energy (J/m)
        self.fieldEnergy = np.zeros(self.Nt)  # electrostatic energy (J/m)
        self.lostE = np.zeros(self.Nt, dtype=int)  # electrons replaced each step
        self.lostI = np.zeros(self.Nt, dtype=int)
        self.mgCycles = np.zeros(self.Nt, dtype=int)

    def printStabilityChecks(self):
        Bx, By = self.cusp.getMagneticFieldAtPosition(self.plasma.loceMatrix[:,0], self.plasma.loceMatrix[:,1])
        Bmax = np.sqrt(Bx**2 + By**2).max()
        wpe = np.sqrt(self.n0*qe**2/(eps0*me))
        Te = me*(np.sqrt(2)*self.plasma.vthermale)**2      # kT_e (J) for the sampled spread
        debye = np.sqrt(eps0*Te/(self.n0*qe**2))
        vmax = 4*np.sqrt(2)*self.plasma.vthermale
        print(f" kTe = {Te/qe:.2f} eV, Debye length / dx = {debye/self.cusp.dx:.1f} (want > ~0.5)")
        print(f" w_pe*dt = {wpe*self.dt:.3g}, w_ce*dt (max at load) = {etome*Bmax*self.dt:.3g} (want < ~0.2)")
        print(f" v*dt/dx = {vmax*self.dt/self.cusp.dx:.3g} (want < 1)")
        print(f" macro-particle weight = {self.plasma.weight:.3g} physical particles per metre in z")

    def updateField(self):
        rho = self.plasma.depositCharge(self.cusp)
        self.cusp.calculateElectricField(rho, tol=self.fieldTol)

    def fieldsAtParticles(self):
        """E and B (Np,3) at electron and ion positions."""
        Np = self.Np
        out = []
        for pos in (self.plasma.loceMatrix, self.plasma.lociMatrix):
            idx, w = self.cusp.cicWeights(pos)
            E = np.zeros((Np, 3)); B = np.zeros((Np, 3))
            E[:, 0], E[:, 1] = self.cusp.gatherField(idx, w)
            B[:, 0], B[:, 1] = self.cusp.getMagneticFieldAtPosition(pos[:, 0], pos[:, 1])
            out.append((E, B))
        (Ee, Be), (Ei, Bi) = out
        return Ee, Ei, Be, Bi

    @staticmethod
    def borisPush(v, E, B, qm, dt):
        """
        Boris scheme: half electric kick, magnetic rotation, half electric kick.
        v (Np,3) at t-dt/2 -> returns v at t+dt/2.
        """
        h = qm*dt/2
        vminus = v + h*E
        t = h*B
        s = 2*t/(1 + np.sum(t*t, axis=1, keepdims=True))
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)
        return vplus + h*E

    def runSimulation(self):
        # Leapfrog: positions at t_n, velocities at t_{n-1/2}.
        # Each step: gather E, B -> Boris push -> move -> walls -> deposit -> solve Poisson
        dt = self.dt
        pl = self.plasma
        wq = pl.weight        # physical particles per macro-particle (per metre in z)
        ts = time.time()
        report = max(1, self.Nt//10)
        for k in range(self.Nt):
            Ee, Ei, Be, Bi = self.fieldsAtParticles()
            vOldE, vOldI = pl.veleMatrix, pl.veliMatrix
            pl.veleMatrix = self.borisPush(vOldE, Ee, Be, self.qme, dt)
            pl.veliMatrix = self.borisPush(vOldI, Ei, Bi, self.qmi, dt)

            # diagnostics at t_k (kinetic energy from the average of v^2 at k-1/2, k+1/2)
            self.kineticE[k] = 0.25*me*wq*np.sum(vOldE**2 + pl.veleMatrix**2)
            self.kineticI[k] = 0.25*me*miontoelec*wq*np.sum(vOldI**2 + pl.veliMatrix**2)
            self.fieldEnergy[k] = 0.5*eps0*np.sum(self.cusp.Ex**2 + self.cusp.Ey**2)*self.cusp.dx*self.cusp.dy
            self.vesq[k] = np.sum(pl.veleMatrix[0]**2)
            self.mgCycles[k] = self.cusp.mgCycles

            # advance particles (2D in x,y; vz is kept for the magnetic rotation)
            pl.loceMatrix += dt*pl.veleMatrix[:, :2]
            pl.lociMatrix += dt*pl.veliMatrix[:, :2]
            lostE, lostI = self.cusp.checkBoundaries(pl)
            self.lostE[k] = lostE.sum()
            self.lostI[k] = lostI.sum()

            # store sample trajectory (NaN gap where the particle was replaced)
            if not lostE[0]:
                self.trajectory[k+1, :] = pl.loceMatrix[0, :]
            if not lostI[0]:
                self.trajectoryi[k+1, :] = pl.lociMatrix[0, :]

            # electric field from the new charge positions
            self.updateField()

            if (k+1) % report == 0:
                print(f" step {k+1}/{self.Nt}  t = {(k+1)*dt*1e9:.2f} ns  "
                      f"replaced e/i so far = {self.lostE[:k+1].sum()}/{self.lostI[:k+1].sum()}  "
                      f"elapsed {time.time()-ts:.1f} s")
        print(" simulation time (s) ", time.time()-ts)

    def plotTrajectory(self):
        print (" Number of time steps ", self.Nt)
        print ("initial e pos ", self.trajectory[0,:])
        print (" final e pos ", self.plasma.loceMatrix[0,:])
        print ("initial i pos ", self.trajectoryi[0,:])
        print (" final i pos ", self.plasma.lociMatrix[0,:])
        fig, axs = plt.subplots(1, 2, figsize=(11, 5))
        for ax, tr, name in ((axs[0], self.trajectory, "electron 0"),
                             (axs[1], self.trajectoryi, "ion 0")):
            self._drawChamber(ax)
            ax.plot(1e3*tr[:, 0], 1e3*tr[:, 1], lw=0.8)
            ax.plot(1e3*tr[0, 0], 1e3*tr[0, 1], "go", label="start")
            ax.set_title(f"Sample trajectory: {name}")
            ax.legend(loc="upper right")
        fig.tight_layout()
        return fig

    def plotFields(self):
        """Final potential with particle positions and magnetic field lines."""
        c = self.cusp
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(self.cusp.phi.T, origin="lower", extent=[0, 1e3*c.LX, 0, 1e3*c.LY],
                       cmap="RdBu_r", vmin=-np.abs(c.phi).max(), vmax=np.abs(c.phi).max())
        fig.colorbar(im, ax=ax, label="potential (V)")
        X, Y = np.meshgrid(c.x[1:], c.y[1:])
        Bx, By = c.getMagneticFieldAtPosition(X, Y)
        ax.streamplot(1e3*X, 1e3*Y, Bx, By, color="0.4", density=1.0, linewidth=0.5, arrowsize=0.6)
        ax.plot(1e3*self.plasma.loceMatrix[:, 0], 1e3*self.plasma.loceMatrix[:, 1], ".",
                ms=1.5, color="tab:orange", label="electrons")
        ax.plot(1e3*self.plasma.lociMatrix[:, 0], 1e3*self.plasma.lociMatrix[:, 1], ".",
                ms=1.5, color="tab:green", label="ions")
        self._drawChamber(ax)
        ax.set_title(f"Potential and particles at t = {self.simTime*1e9:.1f} ns")
        ax.legend(loc="upper right", markerscale=6)
        fig.tight_layout()
        return fig

    def plotEnergy(self):
        t = 1e9*self.time
        fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        axs[0].plot(t, self.kineticE, label="electron kinetic")
        axs[0].plot(t, self.kineticI, label="ion kinetic")
        axs[0].plot(t, self.fieldEnergy, label="electrostatic")
        axs[0].set_ylabel("energy (J/m)")
        axs[0].legend()
        axs[1].plot(t, np.cumsum(self.lostE), label="electrons replaced")
        axs[1].plot(t, np.cumsum(self.lostI), label="ions replaced")
        axs[1].set_ylabel("cumulative count")
        axs[1].set_xlabel("time (ns)")
        axs[1].legend()
        fig.tight_layout()
        return fig

    def _drawChamber(self, ax):
        ax.plot([0, 1e3*self.Lx, 1e3*self.Lx, 0, 0], [0, 0, 1e3*self.Ly, 1e3*self.Ly, 0], "k-", lw=1)
        ax.plot([0], [0], "ko", ms=7); ax.plot([1e3*self.Lx], [0], "kx", ms=8, mew=2)   # wires: +I, -I
        ax.set_xlim(-0.3, 1e3*self.Lx + 0.3); ax.set_ylim(-0.3, 1e3*self.Ly + 0.3)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    def runSimulationCollsions(self):
        # same as above but with collisions
        return None
########

class Plasma():
    """
    Electron and ion macro-particles. Positions (Np,2) in metres, velocities
    (Np,3) in m/s. Each macro-particle stands for `weight` physical particles
    per metre in z, chosen so the loading region has density n0.
    """
    def __init__(self,Np=1000,Lx=0.01,Ly=0.01,n0=1e14,vthermale=343400,vthermali=42775,seed=0):
        # thermal vel in m/s
        self.Np = Np
        self.vthermale = vthermale
        self.vthermali = vthermali
        self.rng = np.random.default_rng(seed)
        self.loadFraction = 0.75     # particles loaded in y in [0.75 Ly, Ly]
        loadArea = Lx*(1 - self.loadFraction)*Ly
        self.weight = n0*loadArea/Np
        self.loadParticles(Np,Lx,Ly)

    def samplePositions(self, n, Lx, Ly):
        p = np.empty((n, 2))
        p[:, 0] = self.rng.uniform(0, Lx, n)
        p[:, 1] = self.rng.uniform(self.loadFraction*Ly, Ly, n)
        return p

    def sampleVelocities(self, n, vth):
        return np.sqrt(2)*vth*self.rng.normal(size=(n, 3))

    def loadParticles(self,Np,Lx,Ly):
        # initialize velocities from a Maxwell distribution
        # initialize positions uniformly in the upper quarter of the chamber.
        # Ions start at the electron positions so the initial charge density is zero.
        self.veleMatrix = self.sampleVelocities(Np, self.vthermale)
        self.veliMatrix = self.sampleVelocities(Np, self.vthermali)
        if self.veleMatrix[0,1] > 0:
            self.veleMatrix[0,1] = -self.veleMatrix[0,1]
        if self.veleMatrix[0,0] > 0:
            self.veleMatrix[0,0] = -self.veleMatrix[0,0]
        self.veleMatrix[0,2] = 0.0
        self.loceMatrix = self.samplePositions(Np, Lx, Ly)
        self.lociMatrix = self.loceMatrix.copy()

    def depositCharge(self, cusp):
        """Charge density rho (C/m^3) on the grid nodes by cloud-in-cell weighting."""
        nn = (cusp.Nx+1)*(cusp.Ny+1)
        idx, w = cusp.cicWeights(self.lociMatrix)
        ni = np.bincount(idx.ravel(), weights=w.ravel(), minlength=nn)
        idx, w = cusp.cicWeights(self.loceMatrix)
        ne = np.bincount(idx.ravel(), weights=w.ravel(), minlength=nn)
        rho = qe*self.weight*(ni - ne)/(cusp.dx*cusp.dy)
        return rho.reshape(cusp.Nx+1, cusp.Ny+1)


def main():
    print("test sim")
    sim = PICSimulation(T=30e-9, Np=20000)
    sim.runSimulation()
    sim.plotTrajectory()
    sim.plotFields()
    sim.plotEnergy()
    plt.show()

if __name__ == "__main__":
    main()
