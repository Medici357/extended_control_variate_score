"""Phase-shifting QPI output module: ray traces and numerical answers.
Units: mm.  Folded geometry (as in the SLIM-type module of the notes):
image plane (z=0) -> L1 (z=f1) -> reflective SLM (z=2 f1) -> back to the
beamsplitter (z=zBS) -> reflected UP -> L2 (f1 path after the SLM) -> CCD (f1 after L2).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- data ----------------
lam  = 500e-6          # mm  (500 nm)
f1   = 33.3            # mm
Dim  = 10.0            # mm, 1/e^2 diameter at image plane
zL1, zSLM = f1, 2*f1
zBS  = 1.5*f1          # beamsplitter centre (assumed; does not affect results)
a    = zSLM - zBS      # SLM -> BS path
YL2  = f1 - a          # BS -> L2 path (so SLM -> L2 path = f1)
YCCD = YL2 + f1        # L2 -> CCD = f1
x1   = 0.10*f1         # object point height for part 2

# ---------------- ray tracing ----------------
def thin_lens(p, d, c, ax, t, f):
    """thin lens: centre c, optical axis ax, transverse unit vector t, focal length f"""
    h = np.dot(p - c, t)
    m = np.dot(d, t)/np.dot(d, ax)
    m2 = m - h/f
    nd = ax + m2*t
    return nd/np.linalg.norm(nd)

def trace(p0, d0):
    p, d = np.array(p0, float), np.array(d0, float)
    d /= np.linalg.norm(d)
    pts = [p.copy()]
    # to L1 (plane z = zL1)
    p = p + (zL1 - p[0])/d[0]*d;  pts.append(p.copy())
    d = thin_lens(p, d, np.array([zL1, 0.]), np.array([1., 0.]), np.array([0., 1.]), f1)
    # to SLM (plane z = zSLM), reflect
    p = p + (zSLM - p[0])/d[0]*d;  pts.append(p.copy())
    d = np.array([-d[0], d[1]])
    # to beamsplitter plane  y = -(z - zBS)  (the "\" diagonal), reflect upward
    t = -((p[0]-zBS) + p[1])/(d[0] + d[1])
    p = p + t*d;  pts.append(p.copy())
    u = np.array([1., -1.])/np.sqrt(2)
    d = 2*np.dot(d, u)*u - d
    # to L2 (plane y = YL2)
    p = p + (YL2 - p[1])/d[1]*d;  pts.append(p.copy())
    d = thin_lens(p, d, np.array([zBS, YL2]), np.array([0., 1.]), np.array([1., 0.]), f1)
    # to CCD (plane y = YCCD)
    p = p + (YCCD - p[1])/d[1]*d;  pts.append(p.copy())
    return np.array(pts), d

def draw(rays, title, fname, src=None):
    fig, ax = plt.subplots(figsize=(8.5, 8.2))
    cols = ['tab:red', 'tab:green', 'tab:blue']
    for (pts, _), c in zip(rays, cols):
        ax.plot(pts[:, 0], pts[:, 1], '-', color=c, lw=1.4)
    # components
    ax.plot([0, 0], [-15, 15], 'k-', lw=1);                      ax.text(0, -18.5, 'image\nplane', ha='center', fontsize=9)
    ax.annotate('', (zL1, 15), (zL1, -15), arrowprops=dict(arrowstyle='<->'));  ax.text(zL1, -18.5, '$L_1$', ha='center')
    ax.plot([zSLM, zSLM], [-14, 14], 'k-', lw=4);                ax.text(zSLM, 17, 'SLM', ha='center')
    ax.plot([zBS + 20, zBS - 20], [-20, 20], color='tab:gray', lw=3);      ax.text(zBS - 22, 22, 'BS', ha='right')
    ax.annotate('', (zBS + 20, YL2), (zBS - 20, YL2), arrowprops=dict(arrowstyle='<->'));  ax.text(zBS + 22, YL2, '$L_2$', va='center')
    ax.plot([zBS - 5.24, zBS + 5.24], [YCCD, YCCD], 'k-', lw=4);   ax.text(zBS + 8, YCCD, 'CCD', va='center')
    ax.axhline(0, color='0.7', lw=0.5, ls='--')
    if src is not None:
        ax.plot(*src, 'ko', ms=4)
    for z in [0, zL1, zSLM]:
        pass
    ax.set_aspect('equal'); ax.set_xlim(-8, 88); ax.set_ylim(-26, 58)
    ax.set_xlabel('$z$ (mm)'); ax.set_ylabel('$y$ (mm)'); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)

# ---------------- Part 1: collimated reference rays ----------------
ref = [trace([0, y], [1, 0]) for y in (+Dim/2, 0.0, -Dim/2)]
draw(ref, 'Part 1: collimated reference rays ($y=+5,\\,0,\\,-5$ mm at the image plane)', 'fig_ref_rays.png')
zc = np.array([r[0][-1, 0] - zBS for r in ref])      # transverse coordinate at CCD (z - zBS)
DCCD = zc.max() - zc.min()
print('Part 1 transverse positions at CCD (mm):', zc, ' exit directions:', [np.round(r[1], 6) for r in ref])

# ---------------- Part 2: point-source rays ----------------
th0 = np.arctan2(-x1, f1)                              # middle ray through centre of L1
angs = [th0 + np.radians(13.35), th0, th0 - np.radians(13.35)]
pt = [trace([0, x1], [np.cos(t), np.sin(t)]) for t in angs]
draw(pt, 'Part 2: rays from the image point $x_1=0.10f_1=3.33$ mm', 'fig_point_rays.png', src=(0, x1))
print('middle-ray angle (deg):', np.degrees(th0))
print('Part 2 CCD hit (z-zBS) :', [round(r[0][-1, 0] - zBS, 4) for r in pt], ' y:', [round(r[0][-1, 1], 4) for r in pt])
print('Part 2 heights on L1 (mm):', [round(r[0][1, 1], 3) for r in pt])

# ---------------- Part 3: ray-transfer matrix (unfolded 4f) ----------------
T  = lambda d: np.array([[1, d], [0, 1]])
L  = lambda f: np.array([[1, 0], [-1/f, 1]])
M  = T(f1) @ L(f1) @ T(f1) @ T(f1) @ L(f1) @ T(f1)     # image plane -> CCD
print('System matrix image plane -> CCD:\n', np.round(M, 9))
print('Part 3: D_CCD (matrix) = %.3f mm ; D_CCD (traced) = %.3f mm' % (abs(M[0, 0])*Dim, DCCD))

# ---------------- Part 4: pixel size ----------------
Npix = 5e6
A = np.pi*DCCD**2/4
dpix = np.sqrt(A/Npix)*1e3
print('Part 4: illuminated area = %.3f mm^2 ; d_pixel = %.3f um' % (A, dpix))
print('        7 MP array side = %.2f mm' % (np.sqrt(7e6)*dpix*1e-3))

# ---------------- Part 5: Gaussian beam at the SLM ----------------
w0 = Dim/2
zR = np.pi*w0**2/lam
q0 = 1j*zR
Msys = T(f1) @ L(f1) @ T(f1)                            # image plane -> SLM (via L1)
q1 = (Msys[0, 0]*q0 + Msys[0, 1])/(Msys[1, 0]*q0 + Msys[1, 1])
zR1 = q1.imag
w1 = np.sqrt(lam*zR1/np.pi)
print('Part 5: z_R(image) = %.2f m ; q_SLM = %s mm ; z_R(SLM) = %.4e mm' % (zR*1e-3, np.round(q1, 9), zR1))
print('w_SLM = %.4f um ; D_Gauss = %.3f um ; check lam f/(pi w0)=%.4f um'% (w1*1e3, 2*w1*1e3, lam*f1/(np.pi*w0)*1e3))