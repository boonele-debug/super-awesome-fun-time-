import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from sklearn.preprocessing import StandardScaler
    import matplotlib.patches as patches
    import numpy as np
    import pandas as pd
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, ConstantKernel as C
    from scipy.stats import entropy
    from scipy.linalg import solve
    return ExtraTreesClassifier, entropy, mo, np, pd, plt, solve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::radix-icons:box:: define search space
    """)
    return


@app.cell
def _():
    # list of components
    components = ["[surfactant] (g/L)", "[salt] (g/L)"]

    # maximum concentrations defining the box
    c_max = [20.0, 40.0] # g/L
    return c_max, components


@app.cell
def _(components, plt):
    def draw_box(c_max):
        fig, ax = plt.subplots()

        plt.xlabel(components[0])
        plt.ylabel(components[1])

        ax.set_aspect('equal', 'box')

        plt.xlim([0, c_max[0]])
        plt.ylim([0, c_max[1]])

        return ax
    return (draw_box,)


@app.cell
def _(c_max, draw_box):
    draw_box(c_max)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::lucide:bow-arrow:: experimental design (a ray)
    """)
    return


@app.cell
def _(np):
    # give a ratio := c1 / c0 and the box boundary
    #   solve for the point on the top or right side of the box
    #   which is the concentration vector for initializing the dilution experiment
    def ray_start_c(c1_c0_ratio, c_max):
        if np.isclose(c1_c0_ratio, 0.0):
            return np.array([c_max[0], 0.0])
        
        t_edge = min(c_max[0] / 1.0, c_max[1] / c1_c0_ratio)
        return np.array([t_edge, t_edge * c1_c0_ratio])
    return (ray_start_c,)


@app.cell
def _(c_max, np, ray_start_c):
    # draw the ray that defines an expt design
    def draw_ray(ax, c1_c0_ratio, draw_start_point=True):
        # draw ray
        c0s = np.linspace(0, c_max[0], 10)
        ax.plot(c0s, c1_c0_ratio * c0s, color="gray", zorder=1, linewidth=1, linestyle="--")

        # dray starting point
        if draw_start_point:
            c_end = ray_start_c(c1_c0_ratio, c_max)
            ax.scatter(c_end[0], c_end[1], color="gray", clip_on=False)
    return (draw_ray,)


@app.cell
def _(c_max, draw_box, draw_ray, draw_toy_boundary, plt):
    _ax = draw_box(c_max)
    _c1_c0_ratio = 3.0
    draw_toy_boundary(_ax)
    draw_ray(_ax, _c1_c0_ratio)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::material-icon-theme:test-js:: toy phase boundary and data for testing
    """)
    return


@app.cell
def _(np):
    # input: c0. output: c1 at phase boundary.
    #   so (c0, c1) is point on the phase boundary
    def toy_phase_boundary(c0):
         return 20 / (1 + np.exp(0.5 * (c0 - 10))) - 0.5 # made-up
    return (toy_phase_boundary,)


@app.cell
def _(c_max, np, toy_phase_boundary):
    def draw_toy_boundary(ax, color="black"):
        c0s = np.linspace(0, c_max[0], 100)
        c1s = toy_phase_boundary(c0s)

        ax.plot(c0s, c1s, color=color, label="toy phase boundary")
    return (draw_toy_boundary,)


@app.cell
def _(c_max, draw_box, draw_toy_boundary, plt):
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)
    plt.show()
    return


@app.cell
def _(toy_phase_boundary):
    # simulates experiment at input concentration vector c
    #  returns true or false.
    #   true = dissolved
    #   false = precipitated
    def simulate_expt(c):
        return int(c[1] < toy_phase_boundary(c[0]))
    return (simulate_expt,)


@app.cell
def _(np, ray_start_c, simulate_expt):
    # returns the two concentration vectors bracketing the phase boundary
    def run_expt(c1_c0_ratio, c_max, n_res=35):
        c_init = ray_start_c(c1_c0_ratio, c_max)

        # scalars to multiply vector c_init
        ts = np.sort(np.random.random(n_res))[::-1] # start at concentrated solution

        crosssed_phase_boundary = True
        for i, t in enumerate(ts):
            # the concentration for this experiment
            c = t * c_init

            # do the experiment
            outcome = simulate_expt(c) # True if dissolved; False if precipitated

            # if dissolved, we crossed the phase boundary
            if outcome:
                crosssed_phase_boundary = True
                return ts[i-1] * c_init, c
    return (run_expt,)


@app.cell
def _(c_max, run_expt):
    run_expt(1/3, c_max, n_res=25)
    return


@app.cell
def _(components, pd, run_expt):
    def run_expts(c1_c0_ratios, c_max, n_res=50):
        data_rows = []
        for c1_c0_ratio in c1_c0_ratios:
            # simulate experiment and get the two concentration vectors bracketing phase boundary
            c_prec, c_diss = run_expt(c1_c0_ratio, c_max)

            # make note of dissovled point
            data_rows.append({"dissolved": True, components[0]: c_diss[0], components[1]: c_diss[1]})

            # make note of preceding preciptated point
            data_rows.append({"dissolved": False, components[0]: c_prec[0], components[1]: c_prec[1]})
        
        data = pd.DataFrame(data_rows)
    
        return data
    return (run_expts,)


@app.cell
def _(c_max, np, run_expts):
    # specify an experimental design
    n_rays = 3
    _thetas = np.linspace(0, np.pi / 2, n_rays) 
    c1_c0_ratios = np.tan(_thetas)

    toy_data = run_expts(c1_c0_ratios, c_max)
    toy_data
    return c1_c0_ratios, n_rays, toy_data


@app.cell
def _(
    c1_c0_ratios,
    c_max,
    components,
    draw_box,
    draw_ray,
    draw_toy_boundary,
    n_rays,
    plt,
    toy_data,
):
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)

    for _r in range(n_rays):
        draw_ray(_ax, c1_c0_ratios[_r])

    plt.scatter(
        toy_data.loc[toy_data["dissolved"].values, components[0]], 
        toy_data.loc[toy_data["dissolved"].values, components[1]], 
        label="dissolved", zorder=5, clip_on=False
    )
    plt.scatter(
        toy_data.loc[~toy_data["dissolved"].values, components[0]], 
        toy_data.loc[~toy_data["dissolved"].values, components[1]], 
        label="precipitate", zorder=5, clip_on=False
    )

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::devicon:dropwizard:: train GP model
    """)
    return


@app.cell
def _(toy_data):
    toy_data
    return


@app.cell
def _(ExtraTreesClassifier, components):
    def fit_surrogate_model(data):
        X = data[components].values
        y = data["dissolved"].values

        rf = ExtraTreesClassifier(n_estimators=1000, bootstrap=False, criterion="entropy")
        rf.fit(X, y)

        return rf
    return (fit_surrogate_model,)


@app.cell
def _(fit_surrogate_model, toy_data):
    rf = fit_surrogate_model(toy_data)
    return (rf,)


@app.function
#posterior sampling 
def latent_posterior_full(base, Xstar, solve):
    K_star = base.kernel_(base.X_train_, Xstar)            # (n_train, n_test)
    latent_mean = K_star.T.dot(base.y_train_ - base.pi_)   # (n_test,)
    v = solve(base.L_, base.W_sr_[:, None] * K_star)       # (n_train, n_test)
    K_ss = base.kernel_(Xstar, Xstar)                      # (n_test, n_test)
    cov = K_ss - v.T @ v
    return latent_mean, cov


@app.cell
def _(C_scaled, gp, np, plt, solve):
    def viz_surrogate_model(ax, rf, c_max, mode="heatmap", res=50, seed=97330, n_samples=20):
        # make grid of concentration vectors
        c1, c2 = np.meshgrid(np.linspace(0, c_max[0], res), np.linspace(0, c_max[1], res))
        C = np.vstack([c1.reshape(c1.size), c2.reshape(c2.size)]).T
    
        if mode == "heatmap":
            # compute prob of class at each grid point
            y_prob = rf.predict_proba(C)[:, 1].reshape((res, res))
        
            cax = ax.imshow(
                y_prob, cmap=plt.cm.PuOr_r, alpha=0.8,
                extent=(0, c_max[0], 0, c_max[1]), 
                origin='lower',
                vmin=0.0, vmax=1.0, zorder=0
            )
            cb = plt.colorbar(cax, ax=ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cb.set_label("Prob[dissolve]")
        elif mode == "samples":
            mean_f, cov_f = latent_posterior_full(gp.base_estimator_, C_scaled, solve)
            cov_f += 1e-6 * np.eye(cov_f.shape[0])
            rng = np.random.default_rng(seed)
            f_samples = rng.multivariate_normal(mean_f, cov_f, size=n_samples)
            f_samples_grid = f_samples.reshape(n_samples, res, res)
            for i in range(n_samples):
                ax.contour(c1, c2, f_samples_grid[i], levels=[0.5],
                           colors="tab:pink", alpha=0.15, linewidths=1.5)
    return (viz_surrogate_model,)


@app.cell
def _(
    c_max,
    components,
    draw_box,
    draw_toy_boundary,
    plt,
    rf,
    toy_data,
    viz_surrogate_model,
):
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)
    viz_surrogate_model(_ax, rf, c_max, mode="heatmap", seed=3)
    plt.scatter(
        toy_data.loc[toy_data["dissolved"].values, components[0]], 
        toy_data.loc[toy_data["dissolved"].values, components[1]], 
        label="dissolved", zorder=5, clip_on=False
    )
    plt.scatter(
        toy_data.loc[~toy_data["dissolved"].values, components[0]], 
        toy_data.loc[~toy_data["dissolved"].values, components[1]], 
        label="precipitate", zorder=5, clip_on=False
    )
    plt.show()
    return


@app.cell
def _(c_max, entropy, gp, np, plt, ratios, ray_edge_point):
    #entropy stuff

    def prob_and_entropy_along_ray(gp, ratio, c_max, entropy, n_t=200):
        c_edge = ray_edge_point(ratio, c_max)
        ts = np.linspace(0, 1, n_t)
        pts = ts[:, None] * c_edge[None, :]
        probs = gp.predict_proba(pts)[:, 1]  #prob dissolve
        p_clean= np.clip(probs, 1e-12, 1.0 - 1e-12)
        H = -(p_clean * np.log(p_clean) + (1.0 - p_clean) * np.log(1.0 - p_clean))
        return ts, probs, H
    query_ratio = ratios[len(ratios) // 2]  # pick a middle ray to inspect

    _ts, _probs, _H = prob_and_entropy_along_ray(gp, query_ratio, c_max, entropy)

    _fig, _axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    _axes[0].plot(_ts, _probs, color="tab:blue")
    _axes[0].set_ylabel("P(dissolve)")
    _axes[0].set_title(f"ratio (salt:surfactant) = {query_ratio:.3f}")
    _axes[0].axhline(0.5, color="gray", ls="--", lw=1)

    _axes[1].plot(_ts, _H, color="tab:red")
    _axes[1].set_ylabel("entropy (nats)")
    _axes[1].set_xlabel("dilution rate t   (0 = origin, 1 = box edge)")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(gp, np, res, solve, xx):
    _mean_f, _cov_f = latent_posterior_full(gp.base_estimator_, xx, solve)
    _cov_f += 1e-6 * np.eye(_cov_f.shape[0])  
    _rng = np.random.default_rng(0)
    n_boundary_samples = 20
    f_samples = _rng.multivariate_normal(_mean_f, _cov_f, size=n_boundary_samples)
    f_samples_grid = f_samples.reshape(n_boundary_samples, res, res)
    return


@app.cell
def _(draw_boundary, np, plt, solve):
    #kernel plotting function

    def viz(data, gp, c_max, surrogate_viz_mode="heatmap", res=50,
            n_boundary_samples=20, seed=0, levels=None):
        c_expts, dissolved = data
        ax = draw_boundary(c_max)

        for outcome in [0, 1]:
            label = "dissolved" if outcome == 1 else "precipitate"
            ax.scatter(c_expts[dissolved == outcome, 0], c_expts[dissolved == outcome, 1],
                       label=label, zorder=5)

        c1, c2 = np.meshgrid(np.linspace(0, c_max[0], res), np.linspace(0, c_max[1], res))
        xx = np.vstack([c1.reshape(c1.size), c2.reshape(c2.size)]).T

        if surrogate_viz_mode == "heatmap":
            y_prob = gp.predict_proba(xx)[:, 1].reshape((res, res))
            cax = ax.imshow(y_prob, cmap=plt.cm.PuOr_r, alpha=0.8,
                             extent=(0, c_max[0], 0, c_max[1]), origin='lower',
                             vmin=0.0, vmax=1.0, zorder=0)
            cb = plt.colorbar(cax, ax=ax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            cb.set_label("Prob of dissolving")

        elif surrogate_viz_mode == "samples":
            if levels is None:
                levels = [0]
            mean_f, cov_f = latent_posterior_full(gp.base_estimator_, xx, solve)
            cov_f += 1e-6 * np.eye(cov_f.shape[0])
            rng = np.random.default_rng(seed)
            f_samples = rng.multivariate_normal(mean_f, cov_f, size=n_boundary_samples)
            f_samples_grid = f_samples.reshape(n_boundary_samples, res, res)
            for i in range(n_boundary_samples):
                ax.contour(c1, c2, f_samples_grid[i], levels=levels,
                           colors="tab:pink", alpha=0.15, linewidths=1.5)

        else:
            raise ValueError(f"unknown surrogate_viz_mode {surrogate_viz_mode!r}; "
                              "expected 'heatmap' or 'samples'")

        ax.legend(bbox_to_anchor=(1.1, 1))
        return ax
    return (viz,)


@app.cell
def _(c_max, data, gp, viz):
    #levels change the contour of the kernel graph
    _ax = viz(data, gp, c_max, surrogate_viz_mode="samples", levels=[-1, 0, 1])
    _ax.set_xlim(0, 15)
    _ax.set_ylim(0, 25)
    _ax.figure
    return


@app.cell
def _(np, ray_edge_point):
    #score theta graph

    def score_theta(gp, c_max, n_theta=200, n_s=200):

        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz 

        thetas = np.linspace(0, np.pi / 2, n_theta)
        scores = np.zeros(n_theta)

        for i, theta in enumerate(thetas):
            ratio = np.tan(theta)

            #handle the two edge cases explicitly (theta=0 and theta=pi/2)
            if not np.isfinite(ratio) or ratio > 1e10:
                c_edge = np.array([0.0, c_max[1]])       #theta = 90 deg
            elif ratio == 0:
                c_edge = np.array([c_max[0], 0.0])       #theta = 0
            else:
                c_edge = ray_edge_point(ratio, c_max)    #normal case


            #take abstract value and turn it into graph
            L = np.linalg.norm(c_edge)                  
            u = c_edge / L                                
            s_vals = np.linspace(0, L, n_s)               
            pts = s_vals[:, None] * u[None, :]           

           #GP uncertainty at each point
            _, var = gp.base_estimator_.latent_mean_and_variance(pts)   #GP's own uncertainty at each point
            #reduce noise
            std = np.sqrt(np.clip(var, 0, None))         

            #integrate uncertainty and normalize by length
            scores[i] = trapz(std, s_vals) / L           

        return thetas, scores
    return (score_theta,)


@app.cell
def _(c_max, gp, np, plt, score_theta):
    thetas, scores = score_theta(gp, c_max)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.rad2deg(thetas), scores)
    ax.set_xlabel("theta (degrees)")
    ax.set_ylabel("integrated std / length")
    ax.set_title("experiment design")
    fig.tight_layout()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
