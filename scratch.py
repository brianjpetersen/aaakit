


"""
class BarycentricRationalInterpolator:

    def __init__(self, zj, fj, wj):
        self.zj = numpy.asarray(zj)
        self.fj = numpy.asarray(fj)
        self.wj = numpy.asarray(wj)
        assert self.zj.ndim == self.fj.ndim == self.wj.ndim == 1
        assert self.zj.size == self.fj.size == self.wj.size
        self.M = len(zj)

    def __call__(self, z):
        zj = self.zj
        fj = self.fj
        wj = self.wj
        Cd = z[:, None] - zj[None, :]
        ij, jj = numpy.nonzero(Cd == 0)
        Cd[ij, jj] = 1
        C = 1/Cd
        f = C.dot(wj*fj)/C.dot(wj)
        f[ij] = fj[jj]
        return f

    def __len__(self):
        return self.M

    @functools.cached_property
    def poles(self):
        zj = self.zj
        wj = self.wj
        B = numpy.eye(len(wj) + 1)
        B[0, 0] = 0
        E = numpy.block([
            [0, wj],
            [numpy.ones((self.M, 1)), numpy.diag(zj)]
        ])
        p = scipy.linalg.eigvals(E, B)
        return numpy.real_if_close(p[numpy.isfinite(p)])

    @functools.cached_property
    def residues(self):
        poles = self.poles
        # compute residues via formula for simple poles using L'Hôpital's rule
        # to evaluate the limit
        ## cf: https://en.wikipedia.org/wiki/Residue_(complex_analysis)#Simple_poles
        zj = self.zj
        fj = self.fj
        wj = self.wj
        C = 1/(poles[:, None] - zj[None, :])
        n = C.dot(fj*wj)
        d = (-C**2).dot(wj)
        residues = n/d
        return residues

    def polres(self):
        return self.poles, self.residues

    @classmethod
    def aaa(cls, z, f, tol=1e-13, M=None, Mmax=200):
        inf = numpy.inf
        z = numpy.asarray(z)
        f = numpy.asarray(f)
        assert z.ndim == f.ndim == 1
        assert z.size == f.size
        N = len(z)
        j = []
        e = []
        hist = []
        k = list(range(N))
        fm = f.mean()*numpy.ones(N)
        r = f - fm
        e.append(numpy.linalg.norm(r, inf))
        rtol = tol*numpy.linalg.norm(f, inf)
        hist.append({
            'j': j.copy(),
            'k': k.copy(),
            'e': e.copy(),
            'f': fm.copy(),
            'r': r.copy(),
        })
        if M is not None:
            Mmax = M
        for m in range(Mmax):
            jm = numpy.argmax(abs(r))
            j.append(jm)
            k.remove(jm)
            C = 1/(z[k, None] - z[None, j])
            A = (f[k, None] - f[None, j])*C
            _, _, Vh = numpy.linalg.svd(A)
            wj = Vh[-1, :].conj()
            wj = wj
            n = C.dot(wj*f[j])
            d = C.dot(wj)
            fm = f.copy()
            fm[k] = n/d
            r = f - fm
            e.append(numpy.linalg.norm(r, inf))
            hist.append({
                'j': j.copy(),
                'k': k.copy(),
                'e': e.copy(),
                'f': fm.copy(),
                'r': r.copy(),
            })
            if M is None and e[-1] < rtol:
                break
        else:
            if M is None:
                warnings.warn('Maximum number of iterations reached.')
        jnz = wj != 0
        wj = wj[jnz]
        j = numpy.asarray(j)
        zj = z[j[jnz]]
        fj = f[j[jnz]]
        b = cls(zj, fj, wj)
        b.j = j
        b.k = asarray(k)
        b.fit_history = hist
        return b


def fit_residues(f, frf, poles):
    f = numpy.ravel(f)
    frf = numpy.ravel(frf)
    assert len(f) == len(frf)
    poles = numpy.ravel(poles)
    C = 1/(1j*f[:, None] - poles[None, :])
    residues, _, _, _ = numpy.linalg.lstsq(C, frf, rcond=None)
    return residues


class Mode:

    def __init__(self, pole, residue):
        self.pole = pole
        self.residue = residue

    def __call__(self, f):
        return self.residue/(1j*f - self.pole)

    def __hash__(self):
        return hash((self.pole, self.residue))

    def __repr__(self):
        return f'{self.__class__.__name__}(pole={self.pole}, residue={self.residue})'


class Modes:

    def __init__(self, poles, residues):
        self.poles = numpy.ravel(poles)
        self.residues = numpy.ravel(residues)
        assert len(self.poles) == len(self.residues)
        self.M = len(self.poles)
        self._modes = [Mode(pole, residue) for pole, residue in zip(poles, residues)]

    @classmethod
    def fit_with_poles(cls, f, frf, poles):
        residues = fit_residues(f, frf, poles)
        modes = cls(poles, residues)
        modes.fit_details = {
            'method': 'fit_with_poles',
            'f': f,
            'frf': frf,
        }
        return modes

    @classmethod
    def fit(cls, f, frf, **kwargs):
        aaa = BarycentricRationalInterpolator.aaa(1j*f, frf, **kwargs)
        poles = aaa.poles
        k = numpy.argsort(poles.imag)
        poles = poles[k]
        residues = fit_residues(f, frf, poles)
        modes = cls(poles, residues)
        modes.fit_details = {
            'method': 'fit',
            'aaa': aaa,
            'f': f,
            'frf': frf,
        }
        return modes

    """
    @staticmethod
    def polish_poles(f, frf, poles):
        f = numpy.ravel(f)
        frf = numpy.ravel(frf)
        assert len(f) == len(frf)
        #absfrf = abs(frf)
        #argfrf = numpy.unwrap(numpy.angle(frf))
        poles = numpy.ravel(poles)
        M = len(poles)
        u = numpy.zeros(2*M)
        u[0::2] = poles.imag
        u[1::2] = poles.real/poles.imag
        lb = []
        ub = []
        for m in range(M):
            #bounds.append((-200, 200))
            lb.append(-numpy.inf)
            ub.append(numpy.inf)
            lb.append(-10)
            ub.append(10)
            #bounds.append((-3, 3))
        def objective(u):
            poles = u[0::2]*u[1::2] + 1j*u[0::2]
            residues = fit_residues(f, frf, poles)
            fit = Modes(poles, residues)
            residual = frf - fit(f)
            return numpy.hstack((residual.real, residual.imag))
        solution = scipy.optimize.least_squares(objective, u, bounds=(lb, ub))
        u = solution.x
        poles = u[0::2]*u[1::2] + 1j*u[0::2]
        return poles
    """

    def __len__(self):
        return self.M

    def __hash__(self):
        return hash((self.poles, self.residues))

    def __iter__(self):
        return iter(self._modes)

    def __getitem__(self, index):
        return self._modes[index]

    def __repr__(self):
        class_name = self.__class__.__name__
        return f'{class_name}(poles={repr(self.poles)}, residues={repr(self.residues)})'

    def __call__(self, f):
        poles = self.poles
        residues = self.residues
        C = 1/(1j*f[:, None] - poles[None, :])
        return C.dot(residues)


def _fit(frf):
    modes = Modes.fit(f, frf, tol=1e-4)
    poles = [
        pole for pole in modes.poles if 0 < pole.imag < 130 and pole.real < 0
    ]
    modes = Modes.fit_with_poles(f, frf, numpy.hstack(([-2, -5, -10, -50, -100, -1e5], poles)))
    return modes


def decompose_modes(frfs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = executor.map(_fit, frfs)
        modes = [future for future in tqdm.tqdm(futures, total=len(frfs))]
    return modes
"""