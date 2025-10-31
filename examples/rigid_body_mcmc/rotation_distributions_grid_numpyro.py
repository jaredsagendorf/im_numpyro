import argparse
import random
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import trimesh

import numpy as np

import scipy as sp
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

import jax.numpy as jnp
from jax import random

import numpyro
from numpyro import handlers, distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.reparam import ProjectedNormalReparam


from quat_utils import quaternion_apply
from pn4 import ProjectedNormal4

CENTERS = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [-1, 1, 1],
    [1, -1, 1],
    [1, 1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [-1, -1, -1]
], dtype=np.float64)
TYPES = ["cube", "octahedron", "tetrahedron"]# "icosohedron", "cylinder"]
COLORS = mpl.color_sequences["tab10"]

def excluded_volume_restraint(X1, X2, radii=0.5, kappa=4.0):
    '''
    This was an attempt at a crude nxn differentibale EV restraint, but didn't work.
    '''
    n1 = len(X1)
    n2 = len(X2)
    i1, i2 = jnp.triu_indices(n1, m=n2)
    d = jnp.linalg.norm(X1[i1] - X2[i2], axis=-1) - 2*radii
    u = -kappa*(d**2)

    return u*(1-jnp.heaviside(d, 0)) 
    
def model(shapes, restraints, translate=True, rotate=True, predictive=False, excluded_volume=True):
    n = len(shapes)
    coords = []
    for i in range(n):
        if shapes[i]["fixed"]:
            # this shape is in a fixed orientation - apply no transform
            x_mv = numpyro.deterministic("X_{}".format(i), shapes[i]["V_targ"])
            coords.append(x_mv)
            continue
        
        if translate:
            #t = numpyro.sample("t{}".format(i), dist.Normal(jnp.zeros(3), shapes[i]["t_scale"]))
            t = numpyro.sample("t{}".format(i), dist.Uniform(-jnp.ones(3), jnp.ones(3)))
        else:
            # no sampled translation - set the shape to a fixed location
            t = shapes[i]["center"]/shapes[i]["t_scale"]
        
        if rotate:
            q = numpyro.sample("q{}".format(i), ProjectedNormal4(jnp.zeros(4)))
            # apply sampled rotation to shape
            x_r = quaternion_apply(q, shapes[i]["V_init"])
        else:
            # no sampled rotation, move to origin
            x_r = shapes[i]["V_targ"] - shapes[i]["center"]
        
        x_mv = numpyro.deterministic("X_{}".format(i), x_r + shapes[i]["t_scale"]*t)
        coords.append(x_mv)
    
    # compute restraint score
    r_obs = []
    for name, r in restraints.items():
        si = r["si"]
        sj = r["sj"]
        di = r["dij"][0]
        dj = r["dij"][1]
        
        if excluded_volume:
            # add excluded volume-like restraint
            #numpyro.factor("ev_{}{}".format(si, sj), excluded_volume_restraint(coords[si], coords[sj]))
            pass # the above didn't seem to work as intended

        if predictive:
            y_ob = None
        else:
            y_ob = r["y_ob"]
        y_pr = numpyro.deterministic("{}_fw".format(name), jnp.linalg.norm(coords[si][di] - coords[sj][dj], axis=-1))
        
        # compute likelihood
        with numpyro.plate("restraint_{}".format(name), len(di)):
            r_obs.append(
                numpyro.sample(name, dist.Normal(y_pr, r["sigma"]), obs=y_ob)
            )
    return r_obs

def make_polyhedron(name, center=True, side_length=1, rand_rotate=False, rand_translate=False, t_scale=1.5):
    if name == "tetrahedron":
        # Define the vertices of the tetrahedron
        vertices = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 1]
        ], dtype=np.float64) * side_length
        edges = np.array([
            [0, 1], [0, 2], [0, 3],
            [1, 2], [1, 3], [2, 3]
        ])
    elif name == "cube":
        vertices = np.array([
            [0, 0, 0],  # Vertex 0
            [1, 0, 0],  # Vertex 1
            [1, 1, 0],  # Vertex 2
            [0, 1, 0],  # Vertex 3
            [0, 0, 1],  # Vertex 4
            [1, 0, 1],  # Vertex 5
            [1, 1, 1],  # Vertex 6
            [0, 1, 1]   # Vertex 7
        ], dtype=np.float64) * side_length
        edges = np.array([
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
        ])
    elif name == "octahedron":
        # Vertices of an octahedron centered at the origin
        vertices = np.array([
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ], dtype=np.float64) * side_length
        edges = np.array([
            (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 4), (2, 5), (3, 4), (3, 5)
        ])
    elif name == "cylinder":
        mesh = trimesh.creation.cylinder(0.5, height=1.0, sections=6)
        vertices = mesh.vertices
        edges = mesh.edges
    elif name == "icosohedron":
        mesh = trimesh.creation.icosahedron()
        vertices = mesh.vertices
        edges = mesh.edges
    vertices -= vertices.mean(axis=0)
    vertices /= 1.5*np.linalg.norm(vertices, axis=-1).max() # scale to fit inside unit box

    if rand_rotate:
        M = sp.stats.special_ortho_group.rvs(dim=3)
        vertices = vertices @ M.T  # rotate the tetrahedron
    if rand_translate:
        t = sp.stats.uniform_direction.rvs(dim=3)
        vertices += t_scale*t 
    
    return vertices, edges

def plot_shape(vertices, edges, ax=None, **kwargs):
    """
    Plot a 3D shape defined by vertices and edges.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    lines = []
    if vertices.ndim == 2:
        for e in edges:
            lines.append(vertices[e])
    else:
        for e in edges:
            lines += list(np.unstack(vertices[:,e], axis=0))
    collection = Line3DCollection(lines, **kwargs)

    ax.add_collection(collection)
    return ax

def simulate_data(n, shape0, shape1, sigma=1.0, scale=0.2, align=False):
    X0 = shape0["V_targ"]
    X1 = shape1["V_targ"]
    if align:
        assert len(X0) == len(X1)
        ind0 = np.arange(len(X0), dtype=int)
        ind1 = ind0
    else:
        dm = cdist(X0, X1)
        ind = np.argsort(dm.ravel())
        i0 = ind // dm.shape[1]
        i1 = ind % dm.shape[1]
        #ind0 = np.random.choice(np.arange(len(X0), dtype=int), size=n)
        #ind1 = np.random.choice(np.arange(len(X1), dtype=int), size=n)
    dij  = [i0[0:n], i1[0:n]]

    y_gt = np.linalg.norm(X1[dij[1]] - X0[dij[0]], axis=-1)
    if sigma == "median":
        sigma = scale*np.median(y_gt)
    y_ob = np.random.normal(loc=y_gt, scale=sigma) # noisy observations of distances

    r = {
        "dij": dij,
        "y_gt": jnp.array(y_gt),
        "y_ob": jnp.array(y_ob),
        "sigma": sigma
    }
    return r

def make_shape_set(n, names=None):
    if names is None:
        names = random.choices(TYPES, k=n)
    
    shapes = []
    for i in range(n):
        vinit, e = make_polyhedron(names[i])
        R = Rotation.from_matrix(sp.stats.special_ortho_group.rvs(dim=3))
        vtarg = R.apply(vinit) + CENTERS[i]
        shapes.append({
            "name": names[i],
            "E": e,
            "V_init": jnp.array(vinit),
            "V_targ": jnp.array(vtarg),
            "R_targ": R,
            "center": jnp.array(CENTERS[i]),
            "color": COLORS[i],
            "fixed": False,
            "t_scale": 2.0,
            "size": len(vinit)
        })
    return shapes

def plot_sample(vertices, edges, ax, shapes=None):
    plot_shape(vertices, edges, ax=ax, color="tab:blue", alpha=0.04)
    if shapes is not None:
        for i in range(len(shapes)):
            s = shapes[i]
            plot_shape(s["V_targ"], s["E"], ax=ax, color='r', ls='--')

    ax.set_title("{} posterior draws".format(vertices.shape[0]))
    ax.axis("off")
    ax.set_aspect('equal')
    return ax

def get_pairs(shapes, cutoff=2.0):
    pairs = []
    pairs_iter = itertools.combinations(range(len(shapes)), 2)
    for i,j in pairs_iter:
        d = np.linalg.norm(shapes[i]["center"] - shapes[j]["center"])
        if d <= cutoff + 1e-5:
            pairs.append((i,j))
    return pairs

def concatenate_sample(sample, shapes, n, align=False, thinned_size=100):
    '''
    This function is just constructing a more efficient representation of the sampled
    vertices to make plotting easier.
    '''
    # concat edges
    edges = [shapes[0]["E"]]
    offset = shapes[0]["size"]
    e_offset = [0, offset]
    for i in range(1, len(shapes)):
        edges.append(shapes[i]["E"] + offset)
        offset += shapes[i]["size"]
        e_offset.append(offset)
    edges = np.concatenate(edges, dtype=int)

    # concat transformed vertices
    verts = []
    step = n//thinned_size
    for i in range(len(shapes)):
        verts.append(sample["X_{}".format(i)][::step])
    verts = np.concatenate(verts, axis=1)

    if align:
        V_targ = np.concatenate([s["V_targ"] for s in shapes])
        for k in range(thinned_size):
            r, _ = Rotation.align_vectors(V_targ, verts[:,:,k])
            x = r.apply(verts[:,:,k])
            verts[:,:,k] = x
    return edges, verts, e_offset

def get_predictive_errors(prediction, restraints, standardize=True, data_key="y_ob", pred_key="{}"):
    metrics = {}
    for name, r in restraints.items():
        y_pp = prediction[pred_key.format(name)]
        y_ob = r[data_key]
        err = y_pp - y_ob
        if standardize:
            err = 100*err/y_ob
        rmse = jnp.sqrt(jnp.mean(err**2, axis=0))
        mae = jnp.mean(jnp.abs(err), axis=0)
        metrics[name] = {}
        metrics[name]["rmse"]= rmse
        metrics[name]["mae"] = mae
        metrics[name]["error"] = err
    
    return metrics

def plot_ppd(prediction, restraints, ax):
    def boxplot(Y, ax, y_label, tick_labels, data=None, title=None):
        positions = np.arange(Y.shape[1])
        vplt = ax.violinplot(Y, positions=positions)
        if isinstance(data, list):
            for d, l in data:
                ax.scatter(positions, d, label=l, s=8)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=60)
        ax.set_title(title)
        ax.legend([vplt['bodies'][0]], [y_label])
    
    Y_PPD = []
    Y_FWD = []
    Y_GTR = []
    Y_OBS = []
    labels = []
    for name, r in restraints.items():
        y_pp = prediction[name]
        y_fw = prediction["{}_fw".format(name)]
        y_ob = r["y_ob"]
        y_gt = r["y_gt"]

        Y_PPD.append(y_pp)
        Y_FWD.append(y_fw)
        Y_OBS.append(y_ob)
        Y_GTR.append(y_gt)
        for k in range(len(r["dij"][0])):
            labels.append(
                "{}.{}-{}.{}".format(r["si"], r["dij"][0][k], r["sj"], r["dij"][1][k])
            )
    
    Y_PPD = np.concatenate(Y_PPD, axis=-1)
    Y_FWD = np.concatenate(Y_FWD, axis=-1)
    Y_OBS = np.concatenate(Y_OBS)
    Y_GTR = np.concatenate(Y_GTR)

    #fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    data = [(Y_OBS, "OBS")]
    boxplot(Y_PPD, ax, "PPD", labels, data=data)
    # boxplot(Y_FWD, axes[0,1], "FWD", labels, data=data)

    # metrics_ob_ppd = get_predictive_errors(prediction, restraints, standardize=True, data_key="y_ob")
    # metrics_gt_ppd = get_predictive_errors(prediction, restraints, standardize=True, data_key="y_gt")
    # metrics_ob_fwd = get_predictive_errors(prediction, restraints, standardize=True, data_key="y_ob", pred_key="{}_fw")
    # metrics_gt_fwd = get_predictive_errors(prediction, restraints, standardize=True, data_key="y_gt", pred_key="{}_fw")
    # plot_predictive_errors(metrics_ob_ppd, restraints, axes[1,0], title="y_ob")
    # plot_predictive_errors(metrics_gt_ppd, restraints, axes[2,0], title="y_gt")
    # plot_predictive_errors(metrics_ob_fwd, restraints, axes[1,1], title="y_ob")
    # plot_predictive_errors(metrics_gt_fwd, restraints, axes[2,1], title="y_gt")

def plot_predictive_errors(metrics, restraints, ax, title=None):
    # combine data
    errors = []
    labels = []
    for name in restraints:
        errors.append(metrics[name]["error"].T)
        r = restraints[name]
        for k in range(len(r["dij"][0])):
            labels.append(
                "{}.{}-{}.{}".format(r["si"], r["dij"][0][k], r["sj"], r["dij"][1][k])
            )
    errors = np.concatenate(errors).T

    ax.violinplot(errors)
    ax.axhline(0, c='k')
    ax.axhline(5, ls="--", c="r")
    ax.axhline(-5, ls="--", c="r")
    ax.set_xticklabels(labels, rotation=60)
    ax.set_title(title)

    return ax

def main(args):
    ### Construct a ground-truth set of polyhedron on a grid
    N = 3
    shapes = make_shape_set(N, names=[
        "cube", "tetrahedron", "octahedron"
    ])
    shapes[0]["fixed"] = True # freeze the first polyhedron to fix the coordinate axes

    ### Generate distance data between the shapes
    scale = 0.05
    restraints = {}
    pairs = get_pairs(shapes)
    for i,j in pairs:
        r = simulate_data(args.num_data, shapes[i], shapes[j], scale=scale, sigma="median", align=False)
        r["si"] = i
        r["sj"] = j
        restraints["y_{}{}".format(i,j)] = r
    
    ### Plot the shapes and distance data
    fig = plt.figure(figsize=(12, 4))
    ax0 = fig.add_subplot(121, projection='3d', proj_type='ortho')

    # plot shapes
    for i in range(N):
        plot_shape(shapes[i]["V_targ"], shapes[i]["E"], ax=ax0, color=shapes[i]["color"], lw=2)

    # plot distance data
    lines = []
    for r in restraints.values():
        si = r["si"]
        sj = r["sj"]
        Vi = shapes[si]["V_targ"]
        Vj = shapes[sj]["V_targ"]
        di = r["dij"][0]
        dj = r["dij"][1]
        for k in range(len(di)):
            lines.append((Vi[di[k]], Vj[dj[k]]))
    collection = Line3DCollection(lines, color="k", ls="--", alpha=0.4)
    ax0.add_collection(collection)
    ax0.axis("off")
    ax0.set_aspect('equal')

    ### Do HMC in pyro
    q_reparam = ProjectedNormalReparam()
    reparam_model = handlers.reparam(model, {"q{}".format(i):q_reparam for i in range(N)})
    
    numpyro.render_model(reparam_model, 
        model_args=(shapes, restraints),
        model_kwargs=dict(translate=args.translate, rotate=args.rotate),
        render_distributions=True,
        filename="model.png"
    )

    kernel = NUTS(reparam_model, step_size=1e-5)
    numpyro.set_host_device_count(args.num_chains)
    mcmc = MCMC(
        kernel,
        num_samples=args.num_samples,
        num_warmup=args.num_warmup,
        num_chains=args.num_chains,
    )
    rng_key = random.PRNGKey(8)
    mcmc.run(rng_key, shapes, restraints, translate=args.translate, rotate=args.rotate, extra_fields=('potential_energy',))
    samples = mcmc.get_samples()
    xtra = mcmc.get_extra_fields()

    ### Visualize posterior
    n = args.num_samples*args.num_chains
    edges, verts, edge_offset = concatenate_sample(samples, shapes, n,
        thinned_size=args.thin_size,
        align=False
    )
    ax1 = fig.add_subplot(132, projection='3d', proj_type='ortho')
    plot_sample(verts, edges, ax1, shapes=shapes)

    ### Get posterior predictive distribution
    rng_key, rng_key_ = random.split(rng_key) # don't fully understand the need for this operation
    predictive = Predictive(reparam_model, samples)
    prediction = predictive(rng_key_, shapes, restraints, translate=args.translate, rotate=args.rotate, predictive=True)
    ax3 = fig.add_subplot(133)
    plot_ppd(prediction, restraints, ax3)
    #metrics = get_predictive_errors(prediction, restraints, standardize=True)
    #plot_predictive_errors(metrics, restraints, axes_2d[2])

    plt.tight_layout()
    plt.show()
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HMC with rigid body rotation and translation")
    parser.add_argument("--num_samples", nargs="?", default=1000, type=int)
    parser.add_argument("--num_data", nargs="?", default=6, type=int)
    parser.add_argument("--num_chains", nargs="?", default=4, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=1000, type=int)
    parser.add_argument("--rng_seed", nargs="?", default=0, type=int)
    parser.add_argument("--thin_size", nargs="?", default=100, type=int)
    parser.add_argument("--translate", action="store_true", default=False)
    parser.add_argument("--rotate", action="store_true", default=False)
    args = parser.parse_args()

    main(args)