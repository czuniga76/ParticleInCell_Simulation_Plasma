"""
Geometric multigrid solver for the 2D Poisson equation with Dirichlet points.

Discrete problem on each level (grid spacing h, h = 1 on the finest grid):

    (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4 u[i,j]) / h^2 = f[i,j]

at free points; u is held at prescribed values at fixed points (walls,
electrodes). For a physical problem  lap(phi) = -rho/eps0  on a grid with
spacing dx (= dy), pass  f = -rho * dx**2 / eps0.

Usage
-----
    mg = MultigridPoisson(fixed_mask, fixed_values)   # build hierarchy once
    u, info = mg.solve(f)                             # full multigrid (FMG)
    u, info = mg.solve(f, u0=u_previous)              # warm-started V-cycles

Christian Zuniga / Claude, 2026. Extends laplace_fmg.py to f != 0 and reuse.
"""
import numpy as np


# ----------------------------------------------------------------------------
# Building blocks (all arrays square, size 2^k + 1)
# ----------------------------------------------------------------------------

def _smooth(u, f, h2, free_int, parity_int, iters):
    """Red-black Gauss-Seidel, updating only free interior points."""
    red = free_int & parity_int
    black = free_int & ~parity_int
    ui = u[1:-1, 1:-1]
    for _ in range(iters):
        for sel in (red, black):
            new = 0.25 * (u[:-2, 1:-1] + u[2:, 1:-1] +
                          u[1:-1, :-2] + u[1:-1, 2:] - h2 * f[1:-1, 1:-1])
            np.copyto(ui, new, where=sel)
    return u


def _residual(u, f, h2, free):
    r = np.zeros_like(u)
    r[1:-1, 1:-1] = f[1:-1, 1:-1] - (u[:-2, 1:-1] + u[2:, 1:-1] +
                                     u[1:-1, :-2] + u[1:-1, 2:] -
                                     4.0 * u[1:-1, 1:-1]) / h2
    r[~free] = 0.0
    return r


def _restrict(r):
    """Full-weighting restriction from (2m+1)^2 to (m+1)^2."""
    m = (r.shape[0] - 1) // 2
    rc = np.zeros((m + 1, m + 1))
    c = r[2:-2:2, 2:-2:2]
    n = r[1:-3:2, 2:-2:2]; s = r[3:-1:2, 2:-2:2]
    w = r[2:-2:2, 1:-3:2]; e = r[2:-2:2, 3:-1:2]
    nw = r[1:-3:2, 1:-3:2]; ne = r[1:-3:2, 3:-1:2]
    sw = r[3:-1:2, 1:-3:2]; se = r[3:-1:2, 3:-1:2]
    rc[1:-1, 1:-1] = (4.0 * c + 2.0 * (n + s + w + e) + (nw + ne + sw + se)) / 16.0
    return rc


def _prolong(ec):
    """Bilinear interpolation from (m+1)^2 to (2m+1)^2."""
    m = ec.shape[0] - 1
    e = np.zeros((2 * m + 1, 2 * m + 1))
    e[::2, ::2] = ec
    e[1::2, ::2] = 0.5 * (ec[:-1, :] + ec[1:, :])
    e[::2, 1::2] = 0.5 * (ec[:, :-1] + ec[:, 1:])
    e[1::2, 1::2] = 0.25 * (ec[:-1, :-1] + ec[1:, :-1] + ec[:-1, 1:] + ec[1:, 1:])
    return e


def _coarsen(F, V):
    """
    Coarse point is fixed if its fine twin or one of the twin's four nearest
    neighbours is fixed (keeps thin electrodes on odd rows/columns); its value
    is the mean of those fixed values.
    """
    m = (F.shape[0] - 1) // 2
    fc = F[::2, ::2].copy()
    vsum = np.zeros((m - 1, m - 1))
    cnt = np.zeros((m - 1, m - 1))
    for di, dj in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        f = F[2 + di:2 * m - 1 + di:2, 2 + dj:2 * m - 1 + dj:2]
        v = V[2 + di:2 * m - 1 + di:2, 2 + dj:2 * m - 1 + dj:2]
        vsum += np.where(f, v, 0.0)
        cnt += f
    vc = np.where(fc, V[::2, ::2], 0.0)
    hit = cnt > 0
    fc[1:-1, 1:-1] |= hit
    vc[1:-1, 1:-1] = np.where(hit, vsum / np.maximum(cnt, 1), 0.0)
    return fc, vc


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------

class MultigridPoisson:
    """
    Parameters
    ----------
    fixed : bool array (n0, n1)
        True at Dirichlet points. Points outside the array are treated as
        fixed (the grid is padded to a (2^k+1)^2 grid internally).
    values : float array (n0, n1) or None
        Dirichlet values at fixed points (ignored elsewhere). Default 0.
    nu1, nu2 : pre- and post-smoothing sweeps per level.
    ncoarse : Gauss-Seidel sweeps used as the exact solve on the 5x5 grid.
    """

    def __init__(self, fixed, values=None, nu1=2, nu2=2, ncoarse=50, coarsest=5):
        fixed = np.asarray(fixed, dtype=bool)
        if values is None:
            values = np.zeros(fixed.shape)
        self.shape = fixed.shape
        self.nu1, self.nu2, self.ncoarse = nu1, nu2, ncoarse
        n0, n1 = self.shape
        M = 2 ** int(np.ceil(np.log2(max(n0, n1) - 1))) + 1
        self.M = M

        F = np.ones((M, M), dtype=bool)
        F[:n0, :n1] = fixed
        V = np.zeros((M, M))
        V[:n0, :n1] = np.where(fixed, values, 0.0)

        self.fixed, self.values, self.h2 = [F], [V], [1.0]
        while self.fixed[-1].shape[0] > coarsest:
            fc, vc = _coarsen(self.fixed[-1], self.values[-1])
            self.fixed.append(fc)
            self.values.append(vc)
            self.h2.append(self.h2[-1] * 4.0)
        self.free = [~m for m in self.fixed]
        self.free_int = [fr[1:-1, 1:-1] for fr in self.free]
        self.parity_int = []
        for m in self.fixed:
            i, j = np.indices((m.shape[0] - 2, m.shape[0] - 2))
            self.parity_int.append((i + j) % 2 == 0)
        self.L = len(self.fixed) - 1

    # -- internals -----------------------------------------------------------
    def _smooth(self, l, u, f, iters):
        return _smooth(u, f, self.h2[l], self.free_int[l], self.parity_int[l], iters)

    def _vcycle(self, l, u, f):
        if l == self.L:
            return self._smooth(l, u, f, self.ncoarse)
        self._smooth(l, u, f, self.nu1)
        r = _residual(u, f, self.h2[l], self.free[l])
        fc = _restrict(r)
        fc[self.fixed[l + 1]] = 0.0
        ec = np.zeros_like(fc)                   # correction is 0 at fixed points
        self._vcycle(l + 1, ec, fc)
        u += _prolong(ec) * self.free[l]
        self._smooth(l, u, f, self.nu2)
        return u

    def _pad(self, a):
        p = np.zeros((self.M, self.M))
        n0, n1 = self.shape
        p[:n0, :n1] = a
        return p

    def residual(self, u_pad, f_pad):
        return np.abs(_residual(u_pad, f_pad, 1.0, self.free[0])).max()

    # -- public ----------------------------------------------------------------
    def solve(self, f, u0=None, tol=1e-6, max_cycles=50):
        """
        Solve A u = f (h = 1 units on the finest grid).

        f   : (n0, n1) right-hand side; values at fixed points are ignored.
        u0  : optional initial guess (e.g. last time step's solution). If
              None, a full-multigrid pass builds the initial guess.
        tol : stop when max|residual| <= tol * scale, where scale is
              max(max|f|, max|Dirichlet values|).

        Returns u (n0, n1) and a dict with 'cycles' and 'residual'.
        """
        fp = self._pad(f)
        fp[self.fixed[0]] = 0.0
        scale = max(np.abs(fp).max(), np.abs(self.values[0]).max())
        if scale == 0.0:
            return np.zeros(self.shape), {"cycles": 0, "residual": 0.0}

        if u0 is None:
            # restrict f to every level
            fl = [fp]
            for l in range(self.L):
                fc = _restrict(fl[-1])
                fc[self.fixed[l + 1]] = 0.0
                fl.append(fc)
            # coarsest solve, then FMG ascent
            u = self.values[self.L].copy()
            self._smooth(self.L, u, fl[self.L], self.ncoarse)
            for l in range(self.L - 1, -1, -1):
                u = _prolong(u)
                u[self.fixed[l]] = self.values[l][self.fixed[l]]
                self._vcycle(l, u, fl[l])
        else:
            u = self._pad(u0)
            u[self.fixed[0]] = self.values[0][self.fixed[0]]

        res = self.residual(u, fp)
        k = 0
        while res > tol * scale and k < max_cycles:
            self._vcycle(0, u, fp)
            res = self.residual(u, fp)
            k += 1
        n0, n1 = self.shape
        return u[:n0, :n1].copy(), {"cycles": k, "residual": res / scale}
