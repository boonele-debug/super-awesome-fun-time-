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
    import matplotlib.ticker as tck
    import numpy as np
    import pandas as pd
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, Matern, ConstantKernel as C
    from scipy.stats import entropy
    from scipy.linalg import solve
    return C, GaussianProcessRegressor, Matern, mo, np, pd, plt


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
    def draw_toy_boundary(ax, color="white"):
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
            outcome = simulate_expt(c) # True if dissolved; False if precipitate

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
        for id, c1_c0_ratio in enumerate(c1_c0_ratios):
            # simulate experiment and get the two concentration vectors bracketing phase boundary
            c_prec, c_diss = run_expt(c1_c0_ratio, c_max)

            # make note of dissovled point
            data_rows.append({"id": id, "dissolved": True, components[0]: c_diss[0], components[1]: c_diss[1]})

            # make note of preceding preciptated point
            data_rows.append({"id": id, "dissolved": False, components[0]: c_prec[0], components[1]: c_prec[1]})
        
        data = pd.DataFrame(data_rows)
        return data
    return (run_expts,)


@app.cell
def _(c_max, np, run_expts):
    # specify an experimental design
    _thetas = np.linspace(0, np.pi / 2, 4) 
    _thetas = np.array([0, np.pi/2, np.pi/8, np.pi/3, np.pi/6])
    n_rays = np.size(_thetas)
    c1_c0_ratios = np.tan(_thetas)

    toy_data = run_expts(c1_c0_ratios, c_max)
    toy_data
    return c1_c0_ratios, n_rays, toy_data


@app.cell
def _(components):
    def viz_data(ax, data):
        ax.scatter(
            data.loc[data["dissolved"].values, components[0]], 
            data.loc[data["dissolved"].values, components[1]], 
            label="dissolved", zorder=5, clip_on=False, color="aquamarine", 
            edgecolor="black"
        )
    
        ax.scatter(
            data.loc[~data["dissolved"].values, components[0]], 
            data.loc[~data["dissolved"].values, components[1]], 
            label="precipitate", zorder=5, clip_on=False, color="khaki",
            edgecolor="black"
        )
    return (viz_data,)


@app.cell
def _(
    c1_c0_ratios,
    c_max,
    draw_box,
    draw_ray,
    draw_toy_boundary,
    n_rays,
    plt,
    toy_data,
    viz_data,
):
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)

    for _r in range(n_rays):
        draw_ray(_ax, c1_c0_ratios[_r])

    viz_data(_ax, toy_data)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::devicon:dropwizard:: train GP model

    view data as $r=r(\theta)$.
    """)
    return


@app.cell
def _(components, np, pd):
    def build_polar_data(data, c_max):
        # divide by max concentration to normalize (0-1 range)
        c0 = data[components[0]].values / c_max[0]
        c1 = data[components[1]].values / c_max[1]

        # find distance r from origin using pythagorean theorem
        r_all = np.sqrt(c0 ** 2 + c1 ** 2)

        # compute angle c0 and c1 make w/ axis. arctan2 handles case of c0=0
        theta_all = np.arctan2(c1, c0)

        n_rays = len(data) // 2 # data has 2 rows per ray

        polar_rows = []
        for i in range(n_rays):
            r_diss = r_all[2 * i]      # dissolved row
            r_prec = r_all[2 * i + 1]  # precipitated row

            assert data.loc[2 * i, "id"] == data.loc[2 * i + 1, "id"]

            # noise- how far away are my two values on the ray? 
            # 1e-6 is a small enough number to fill in case we get 0, which can cause issues w/ regressor
            alpha_i =  (max(abs(r_prec - r_diss), 1e-6) / 2) ** 2

            # choosing a midpoint for variables because that is most likely where phase boundary will fall
            c0_mid = 0.5 * (c0[2 * i] + c0[2 * i + 1])
            c1_mid = 0.5 * (c1[2 * i] + c1[2 * i + 1])

            # pythagorean theorem and angle with midpoints to estimate where the boundary will fall
            r_mid = np.sqrt(c0_mid**2 + c1_mid**2)
            theta_mid = np.arctan2(c1_mid, c0_mid) # y = c0 (surfactant), x = c1 (salt)

            # build dictionary/new dataframe with new values
            polar_rows.append(
                {
                    "theta": theta_mid,
                    "r": r_mid,
                    "alpha": alpha_i,
                    "r_diss": r_diss,
                    "r_prec": r_prec,
                }
            )

        return pd.DataFrame(polar_rows)
    return (build_polar_data,)


@app.cell
def _(build_polar_data, c_max, toy_data):
    polar_data = build_polar_data(toy_data, c_max)
    polar_data
    return (polar_data,)


@app.cell
def _(C, GaussianProcessRegressor, Matern, np, polar_data):
    kernel = C(1.0, constant_value_bounds=(0.1, 100.0)) * Matern(
        length_scale=0.5, length_scale_bounds=[np.pi/10, np.pi], nu=2.5
    )

    # Initialize GPR without normalize_y to preserve alpha scaling
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=polar_data["alpha"].values,
        normalize_y=True,
        n_restarts_optimizer=10
    )

    # Fit GP model w/ theta and r values from data above
    gpr.fit(polar_data[["theta"]].values, polar_data["r"].values)

    print("Fitted kernel:", gpr.kernel_)
    print("Log-marginal-likelihood:", gpr.log_marginal_likelihood(gpr.kernel_.theta))
    return (gpr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ::lucide:eye:: visualize the surrogate model
    """)
    return


@app.cell
def _(draw_ray, np):
    # plot in data space
    def viz_surrogate_model(
        ax, gpr, c_max, 
        mode="band", res=200, seed=97330,
        n_samples=25, c1_c0_ratios=None
    ):
        # draw rays
        if c1_c0_ratios is not None:
            for c1_c0_ratio in c1_c0_ratios:
                draw_ray(ax, c1_c0_ratio)
            
        # c_max = [c_salt_max, c_surfactant_max] (index order)
        theta_grid = np.linspace(0, np.pi / 2, res)

        #asks gpr for predicted values at every point, flatten for lists (graphing)
        if mode == "band":
            r_mean, r_std = gpr.predict(theta_grid.reshape(-1, 1), return_std=True)
            r_mean = r_mean.flatten()
            r_std = r_std.flatten()

            # top and bottom edge of uncertainty band, bottom will not go below 0
            r_up = r_mean + r_std
            r_lo = np.clip(r_mean - r_std, 0, None)

            # convert polar coordinates back to cartesian
            x_up = r_up * np.cos(theta_grid) * c_max[0]
            y_up = r_up * np.sin(theta_grid) * c_max[1] 

            x_lo = r_lo * np.cos(theta_grid) * c_max[0]
            y_lo = r_lo * np.sin(theta_grid) * c_max[1]

            # confidence band
            ax.fill(
                np.concatenate([x_up, x_lo[::-1]]),
                np.concatenate([y_up, y_lo[::-1]]),
                color="orchid",
                alpha=0.25,
                zorder=3,
                label="±1 std"
            )
        
            # mean
            x = r_mean * np.cos(theta_grid) * c_max[0]
            y = r_mean * np.sin(theta_grid) * c_max[1]
            ax.plot(x, y, color="orchid", linewidth=2, zorder=4, label="GP mean boundary")

        # asks model to generate 25 samples instead of one mean guess
        elif mode == "samples":
            r_samples = gpr.sample_y(theta_grid.reshape(-1, 1), n_samples=n_samples, random_state=seed)
            for i in range(n_samples):
                r = np.clip(r_samples[:, i], 0, None)
                x = r * np.cos(theta_grid) * c_max[0]
                y = r * np.sin(theta_grid) * c_max[1]
                ax.plot(
                    x, y, 
                    color="tab:pink", 
                    alpha=0.2, 
                    linewidth=1.5, 
                    zorder=3
                )
        else:
            raise ValueError(f"unknown mode {mode!r}; expected 'band' or 'samples'")
    return (viz_surrogate_model,)


@app.cell
def _(
    c1_c0_ratios,
    c_max,
    draw_box,
    draw_toy_boundary,
    gpr,
    plt,
    toy_data,
    viz_data,
    viz_surrogate_model,
):
    # build graph
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)
    viz_surrogate_model(_ax, gpr, c_max, mode="samples", c1_c0_ratios=c1_c0_ratios)
    viz_data(_ax, toy_data)
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.show()
    return


@app.cell
def _(
    c1_c0_ratios,
    c_max,
    draw_box,
    draw_toy_boundary,
    gpr,
    plt,
    toy_data,
    viz_data,
    viz_surrogate_model,
):
    # build graph
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)
    viz_surrogate_model(_ax, gpr, c_max, mode="band", c1_c0_ratios=c1_c0_ratios)
    viz_data(_ax, toy_data)
    plt.legend(bbox_to_anchor=(1.1, 1))
    plt.show()
    return


@app.cell
def _(np, plt):
    def viz_surrogate_model_polar(gpr, polar_data):
        fig, ax = plt.subplots()
    
        plt.xlabel(r"$\theta$")
        plt.ylabel("$r$")

        plt.scatter(
            polar_data["theta"], polar_data["r"], clip_on=False,
            color="plum", 
            edgecolor="black", zorder=10
        )
        plt.xlim([0, np.pi/2])

        theta_grid = np.linspace(0, np.pi / 2, 150)

        r_mean, r_std = gpr.predict(theta_grid.reshape(-1, 1), return_std=True)

        # confidence band
        ax.fill_between(
            theta_grid, 
            r_mean - r_std,
            r_mean + r_std,
            color="orchid",
            alpha=0.25,
            zorder=3,
            label="±1 std"
        )
    
        ax.plot(theta_grid, r_mean, color="orchid", linewidth=2, zorder=4, label="GP mean boundary")

        ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
        ax.set_xticklabels([r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
        plt.ylim(ymin=0)
    
        plt.show()
    return (viz_surrogate_model_polar,)


@app.cell
def _(gpr, polar_data, viz_surrogate_model_polar):
    viz_surrogate_model_polar(gpr, polar_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::lucide:target:: active learning — which ray to run next
    """)
    return


@app.cell
def _(components, np, plt):
    def choose_theta(gpr, n_theta=100):
        # candidate theta's
        thetas = np.linspace(0, np.pi / 2, n_theta)

        # GP uncertainty at r at each theta
        _, scores = gpr.predict(thetas.reshape(-1, 1), return_std=True)

        # total integrated uncertanty
        total_integrated_std =  np.trapezoid(scores, x=thetas)

        fig, ax = plt.subplots(figsize=(6, 4))
    
        ax.plot(thetas, scores, color="rebeccapurple")
    
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("posterior std of $r$")
        ax.set_title("acquisition scores")
    
        best_theta = thetas[np.argmax(scores)]
        best_c1_ovr_c0_ratio = np.tan(best_theta)
        ax.axvline(
            best_theta, color="black", ls="--", lw=1,
            label=f"next design:\n$\\theta^*$={best_theta:.1f} rad\n{components[1].split()[0]}: {components[0].split()[0]}: {best_c1_ovr_c0_ratio:.1f}"
        )
    
        best_deg = best_theta
        max_score = np.max(scores)
    
        ax.plot(best_deg, max_score, 'o', color="black", zorder=5)
    
        ax.legend(title=f"total uncertainty: {total_integrated_std:.2f}")
    
        ax.set_xticks([0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2])
        ax.set_xticklabels([r"$0$", r"$\pi/8$", r"$\pi/4$", r"$3\pi/8$", r"$\pi/2$"])
    
        plt.xlim([0, np.pi/2])
        plt.ylim(ymin=0)
    
        fig.tight_layout()
        plt.show()

        return best_c1_ovr_c0_ratio
    return (choose_theta,)


@app.cell
def _(choose_theta, gpr):
    choose_theta(gpr)
    return


if __name__ == "__main__":
    app.run()
