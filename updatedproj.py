import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from sklearn.preprocessing import StandardScaler
    import matplotlib.patches as patches
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


@app.cell
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
    n_rays = 4
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


@app.cell
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
def _(components, np, pd):
    def build_polar_data(data, c_max): #toy data, concentrations
      #pull data from the dataframe, and convert to array for quicker calculations. (.values)
    
        #divide by max concentration to normalize (0-1 range)
        c1 = data[components[0]].values / c_max[1] #salt
        c0 = data[components[1]].values / c_max[0] #surfactant
   
        #find distance r from origin using pythagorean theorem
        r_all = np.sqrt(c0 ** 2 + c1 ** 2)
    
        #compute angle c0 and c1 make w/ axis. arctan2 handles case of c0=0
        theta_all = np.arctan2(c1, c0)

        n_rays = len(data) // 2 #data has 2 rows per ray

        #collect dictionary per ray
        polar_rows = []
        for i in range(n_rays):
            r_diss = r_all[2 * i]      # dissolved row
            r_prec = r_all[2 * i + 1]  # precipitated row

            #noise- how far away are my two values on the ray? 
            #1e-6 is a small enough number to fill in case we get 0, which can cause issues w/ regressor
            alpha_i =  max(abs(r_prec - r_diss), 1e-6)
       
            #choosing a midpoint for variables because that is most likely where phase boundary will fall
            c0_mid = 0.5 * (c0[2 * i] + c0[2 * i + 1])
            c1_mid = 0.5 * (c1[2 * i] + c1[2 * i + 1])

            #pythagorean theorem and angle with midpoints to estimate where the boundary will fall
            r_mid = np.sqrt(c0_mid**2 + c1_mid**2)
            theta_mid = np.arctan2(c1_mid, c0_mid) # y = c0 (surfactant), x = c1 (salt)

           #Build dictionary/new dataframe with new values
            polar_rows.append({
                "theta": theta_mid,
                "r": r_mid,
                "alpha": alpha_i,
                "r_diss": r_diss,
                "r_prec": r_prec,
            })
        return pd.DataFrame(polar_rows)

    return (build_polar_data,)


@app.cell
def _(build_polar_data, c_max, toy_data):
    polar_data = build_polar_data(toy_data, c_max)

    polar_data
    return (polar_data,)


@app.cell
def _(C, GaussianProcessRegressor, Matern, polar_data):
    kernel = C(1.0, constant_value_bounds=(0.1, 100.0)) * Matern(
        length_scale=0.5, length_scale_bounds="fixed", nu=2.5
    )

    #Initialize GPR without normalize_y to preserve alpha scaling
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=polar_data["alpha"].values,
        normalize_y=False,
        n_restarts_optimizer=10
    )

    # Fit GP model w/ theta and r values from data above
    gpr.fit(polar_data[["theta"]].values, polar_data["r"].values)
    return (gpr,)


@app.cell
def _(mo):
    mo.md(r"""
    w1# ::lucide:eye:: visualize the surrogate model
    """)
    return


@app.cell
def _(np):
    def viz_surrogate_model(ax, gpr, c_max, mode="band", res=200, seed=97330,
                             n_samples=25, draw_rays=None):
    
        #c_max = [c_salt_max, c_surfactant_max] (index order)
        theta_grid = np.linspace(0, np.pi / 2, res)
        if draw_rays is not None:
            c0s = np.linspace(0, c_max[0], 10)
            for ratio in draw_rays:
                ax.plot(c0s, ratio * c0s, color="gray", linestyle="--",
                     linewidth=1, alpha=0.6, zorder=0)

        #asks gpr for predicted values at every point, flatten for lists (graphing)
        if mode == "band":
            r_mean, r_std = gpr.predict(theta_grid.reshape(-1, 1), return_std=True)
            r_mean = r_mean.flatten()
            r_std = r_std.flatten()

        #top and bottom edge of uncertainty band, bottom will not go below 0
            r_up = r_mean + r_std
            r_lo = np.clip(r_mean - r_std, 0, None)

        #convert polar coordinates back to cartesian
            x_up = r_up * np.sin(theta_grid) * c_max[1] #salt
            y_up = r_up * np.cos(theta_grid) * c_max[0] #surfactant

            x_lo = r_lo * np.sin(theta_grid) * c_max[1]
            y_lo = r_lo * np.cos(theta_grid) * c_max[0]
        
        #confidence band
            ax.fill(
                np.concatenate([x_up, x_lo[::-1]]),
                np.concatenate([y_up, y_lo[::-1]]),
                color="orchid",
                alpha=0.25,
                zorder=3,
                label="±1 std"
            )

        #same conversion using model's best guess, where boundarxy actually sits
            salt_mean       = r_mean * np.sin(theta_grid) * c_max[1]
            surfactant_mean = r_mean * np.cos(theta_grid) * c_max[0]
            ax.plot(salt_mean, surfactant_mean, color="orchid", linewidth=2, zorder=4, label="GP mean boundary")

       #asks model to generate 25 samples instead of one mean guess
        elif mode == "samples":
            r_samples = gpr.sample_y(theta_grid.reshape(-1, 1), n_samples=n_samples, random_state=seed)
            for i in range(n_samples):
                r_i = np.clip(r_samples[:, i], 0, None)
                salt_i       = r_i * np.sin(theta_grid) * c_max[1]
                surfactant_i = r_i * np.cos(theta_grid) * c_max[0]
                ax.plot(
                    salt_i, 
                    surfactant_i, 
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
    components,
    draw_box,
    draw_toy_boundary,
    gpr,
    plt,
    toy_data,
    viz_surrogate_model,
):
    #build graph
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)
    viz_surrogate_model(_ax, gpr, c_max, mode="band", draw_rays=c1_c0_ratios)

    #take points from dataframe and alternate
    salt_col = components[0]
    surfactant_col = components[1]

    #dissolved is even, start at 0 and pull data from even rows
    salt_dissolved = toy_data[salt_col].values[0::2]
    surfactant_dissolved = toy_data[surfactant_col].values[0::2]

    #precipitate is odd, start at 1 and pull data from odd rows
    salt_precipitate = toy_data[salt_col].values[1::2]
    surfactant_precipitate = toy_data[surfactant_col].values[1::2]

    #plot scatter
    _ax.scatter(
        salt_dissolved, 
        surfactant_dissolved, 
        label="dissolved", 
        color="aquamarine", 
        zorder=5
    )
    _ax.scatter(
        salt_precipitate, 
        surfactant_precipitate, 
        label="precipitate", 
        color="khaki", 
        zorder=5
    )

    #labels and such
    _ax.set_xlabel("[salt] (g/L)")
    _ax.set_ylabel("[surfactant] (g/L)")
    _ax.set_title("Phase Boundary Model")

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
    viz_surrogate_model,
):
    #samples instead of bands
    _ax = draw_box(c_max)
    draw_toy_boundary(_ax)
    viz_surrogate_model(_ax, gpr, c_max, mode="samples", draw_rays=c1_c0_ratios)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    S# ::lucide:target:: active learning — which ray to run next
    """)
    return


@app.cell
def _(np):
    def score_theta(gpr, n_theta=200):
        theta_grid = np.linspace(0, np.pi / 2, n_theta)
        _, r_std = gpr.predict(theta_grid.reshape(-1, 1), return_std=True)
        return theta_grid, r_std

    return (score_theta,)


@app.cell
def _(np):
    #gives a number to show how uncertain the model is
    def integrated_uncertainty(gpr, n_theta=200):
        theta_grid = np.linspace(0, np.pi / 2, n_theta)
        _, r_std = gpr.predict(theta_grid.reshape(-1, 1), return_std=True)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return trapz(r_std, theta_grid)

    return (integrated_uncertainty,)


@app.cell
def _(gpr, integrated_uncertainty):
    integrated_uncertainty(gpr)
    return


@app.cell
def _(gpr, np, plt, score_theta):
    thetas, scores = score_theta(gpr)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.rad2deg(thetas), scores)
    ax.set_xlabel("theta (degrees)")
    ax.set_ylabel("predictive std of r(theta)")
    ax.set_title("experiment design (next-best ray)")

    best_theta = thetas[np.argmax(scores)]
    ax.axvline(np.rad2deg(best_theta), color="gray", ls="--", lw=1,
               label=f"next ray: {np.rad2deg(best_theta):.1f} deg")

    best_deg = np.rad2deg(best_theta)
    max_score = np.max(scores)

    ax.plot(best_deg, max_score, 'o', color="gray", zorder=5)

    ax.legend()
    fig.tight_layout()
    fig
    return (best_theta,)


@app.cell
def _(best_theta, np):
    #convert the recommended angle back into a c1/c0 ratio, ready to hand
    #to run_expt() for the next real dilution series
    suggested_ratio = np.tan(best_theta)
    suggested_ratio
    return


if __name__ == "__main__":
    app.run()
