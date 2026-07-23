import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib.patches as patches
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, ConstantKernel as C
    from scipy.stats import entropy
    from scipy.linalg import solve

    return C, GaussianProcessClassifier, Matern, entropy, np, plt, solve


@app.cell
def _():
    # defines box dims (OG heatmap)
    components = ["surfactant", "salt"]
    c_max = [20.0, 40.0] # g/L
    return (c_max,)


app._unparsable_cell(
    r"""
    def draw_boundary(c_max):
        fig, ax = plt.subplots()
        plt.xlabel("[" + components[0] + "] (g / L)")
        plt.ylabel("[" + components[1] + "] (g / L)")
        ax.set_aspect('equal', 'box')
        plt.xlim([0, c_max[0]])
        plt.ylim([0, c_max[1]])
        c0s = np.linspace(0, c_max[0], 100
        plt.plot(c0s, phase_boundary(c0s), color="white", label="true phase boundary")
        return ax
    """,
    name="_"
)


@app.cell
def _(np):
    #the return statement is what i have been using to put different functions on my phase boundary line. flexible with everything except sine graph. 
    def phase_boundary(c0):
         return 20 / (1 + np.exp(0.5*(c0 - 10)))

    return (phase_boundary,)


@app.cell
def _(phase_boundary):
    def run_expt_point(c):
        return int(c[1] < phase_boundary(c[0]))

    return (run_expt_point,)


@app.cell
def _(np):
    def ray_edge_point(ratio, c_max):
        t_edge = min(c_max[0] / 1.0, c_max[1] / ratio)
        return np.array([t_edge, t_edge * ratio])

    return (ray_edge_point,)


@app.cell
def _(np, ray_edge_point, run_expt_point):
    def run_expt(ratio, c_max, n_max_halvings=25):
        c_edge = ray_edge_point(ratio, c_max)
        t = 1.0
        prev_t = None
        for _ in range(n_max_halvings):
            c = t * c_edge
            outcome = run_expt_point(c)
            if outcome == 1:
                if prev_t is None:
                    # already dissolved at the most concentrated point in the box
                    return np.array([c_edge, c]), np.array([0, 1])
                return np.array([prev_t * c_edge, t * c_edge]), np.array([0, 1])
            prev_t = t
            t /= 2.0
        # never dissolved within n_max_halvings steps
        return np.array([prev_t * c_edge, t * c_edge]), np.array([0, 0])

    return (run_expt,)


@app.cell
def _(c_max, np, run_expt):
    #rays and ratios
    n_rays = 8
    angles_deg = np.linspace(10, 80, n_rays)  
    ratios = np.tan(np.deg2rad(angles_deg))

    #run one dilution series experiment per ray
    _c_expts_list, _dissolved_list = [], []
    for _r in ratios:
        _pts, _labels = run_expt(_r, c_max)
        _c_expts_list.append(_pts)
        _dissolved_list.append(_labels)

    c_expts = np.vstack(_c_expts_list)
    dissolved = np.concatenate(_dissolved_list)
    data = (c_expts, dissolved)
    return c_expts, data, dissolved, ratios


@app.cell
def _(c_max, np):
    #ray cosmetics
    def draw_ray(ax, ratio):
        c0s = np.linspace(0, c_max[0], 10)
        ax.plot(c0s, ratio * c0s, color="gray", zorder=1, linewidth=1)

    return (draw_ray,)


@app.cell
def _(c_expts, c_max, dissolved, draw_boundary, draw_ray, plt, ratios, y_prob):
    #graph cosmetics/inputting the ratio function
    _ax = draw_boundary(c_max)
    for _ratio in ratios:
        draw_ray(_ax, _ratio)

    for _outcome in [0, 1]:
        _label = "dissolved" if _outcome == 1 else "precipitate"
        plt.scatter(c_expts[dissolved == _outcome, 0], c_expts[dissolved == _outcome, 1], label=_label, zorder=5)
    plt.legend(bbox_to_anchor=(1.1, 1))
    cax = plt.imshow(
        y_prob, 
        cmap=plt.cm.PuOr_r, 
        alpha=0.8, 
        extent=(0, c_max[0], 0, c_max[1]),
        origin='lower',  
        vmin=0.0,        
        vmax=1.0,
        zorder=1
    )
    norm = plt.matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)

    cb = plt.colorbar(cax, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], norm=norm)
    cb.set_label(r"Prob of dissolving")
    plt.clim(0, 1)


    plt.show()
    return


@app.cell
def _(c_max, gp, np):
    #lay points
    res = 25
    c1, c2 = np.meshgrid(np.linspace(0, c_max[0], res), np.linspace(0, c_max[1], res))

    #reshape for input to GP
    xx = np.vstack([c1.reshape(c1.size), c2.reshape(c2.size)]).T

    #use GP to predict probability of dissolving, and reshape for heatmap plotting
    y_prob = gp.predict_proba(xx)[:, 1]
    y_prob = y_prob.reshape((res, res)) 
    return res, xx, y_prob


@app.cell
def _(C, GaussianProcessClassifier, Matern, c_expts, dissolved):
    #le kernel
    kernel = C(1.0, constant_value_bounds=(0.1, 100.0)) * Matern(length_scale=[1.0, 1.0], length_scale_bounds=(1.0, 20.0), nu=2.5)

    gp = GaussianProcessClassifier(kernel=kernel)

    #set to none to try and make graph look better
    optimizer=None

    gp.fit(c_expts, dissolved)
    return (gp,)


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


@app.function
#posterior sampling 
def latent_posterior_full(base, Xstar, solve):
    K_star = base.kernel_(base.X_train_, Xstar)           #(n_train, n_test)
    latent_mean = K_star.T.dot(base.y_train_ - base.pi_)   #(n_test,)
    v = solve(base.L_, base.W_sr_[:, None] * K_star)       #(n_train, n_test)
    K_ss = base.kernel_(Xstar, Xstar)                      #(n_test, n_test)
    cov = K_ss - v.T @ v
    return latent_mean, cov


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
