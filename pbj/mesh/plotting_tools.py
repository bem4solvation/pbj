import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly
import plotly.graph_objs as go
import matplotlib.colors as mcolors


def plot_multiple_surfaces(
    simulation,
    values="phi",
    units="kt",
    max_colorbar_scale=1,
    name="plot_multiple_surface.png",
    savefig=True,
    location="ses",
    show_axis=True,
    camera_view=None,
    isosurface_potential_vals=(None, None, None),
    min_max_vals=None,
    figsize=(900, 800),
    solutes_plot=list(),
    internal_derivative_plot=False,
):
    r"""Plot one or more solute surface fields as interactive 3D surfaces.

    The function gathers potential or derivative values from the selected solutes,
    maps them to a continuous colormap, and renders the corresponding triangular
    surface meshes with Plotly. It can optionally save a static image and overlay
    an isosurface based on externally supplied grid data.

    Args:
        simulation (object): Simulation object containing a `solutes` collection.
        values (str, optional): Surface quantity to visualize. Supported values are
            "phi" for the surface potential and "d_phi" for its derivative.
            Defaults to "phi".
        units (str, optional): Unit identifier passed to the solute accessors for
            value conversion. Defaults to "kt".
        max_colorbar_scale (float, optional): Scaling factor used to set the absolute
            colorbar range when `min_max_vals` is not provided. Defaults to 1.
        name (str, optional): Output filename used when `savefig=True`. Defaults to
            "plot_multiple_surface.png".
        savefig (bool, optional): If True, writes the figure to `name` using Plotly's
            image export support. Defaults to True.
        location (str, optional): Surface mesh location to visualize. Use "ses" for the
            standard surface mesh or "stern" for the Stern-layer mesh when available.
            Defaults to "ses".
        show_axis (bool, optional): If True, shows the scene axes in the rendered figure.
            Defaults to True.
        camera_view (tuple, optional): Optional camera eye coordinates for the Plotly scene.
            Defaults to None.
        isosurface_potential_vals (tuple, optional): Optional tuple of the form
            `(coordinates, potential, isovalue)` used to add an isosurface overlay.
            Defaults to `(None, None, None)`.
        min_max_vals (tuple, optional): Optional `(min_value, max_value)` pair used to
            override the automatic colorbar scale. Defaults to None.
        figsize (tuple, optional): Width and height of the rendered figure in pixels.
            Defaults to `(900, 800)`.
        solutes_plot (list, optional): List of solute indices or solute names to include in
            the plot. Defaults to an empty list.
        internal_derivative_plot (bool, optional): If True, requests the internal
            derivative form when `values="d_phi"`. Defaults to False.

    Returns:
        None

    Side Effects:
        - Renders an interactive Plotly figure in the browser.
        - Optionally writes a static image to disk when `savefig` is True.
        - Updates the figure layout and colorbar based on the selected surface values.
    """

    if not min_max_vals:
        vals_max = []
        for solute in simulation.solutes:
            if values == "phi":
                phi_vertices, unit_label = solute.get_surface_potential(
                    units=units, print_units=False
                )
            elif values == "d_phi":
                phi_vertices, unit_label = solute.get_surface_potential_derivative(
                    units=units,
                    print_units=False,
                    internal_derivative=internal_derivative_plot,
                )
            phi_simplices = np.mean(phi_vertices[solute.mesh.elements.T], axis=1)
            vals_max.append(max(abs(np.max(phi_simplices)), abs(np.min(phi_simplices))))
        val_max = max_colorbar_scale * max(vals_max)
        val_min = -val_max
        cmap1 = plt.get_cmap("autumn_r", 128)
        cmap2 = plt.get_cmap("bwr_r", 128)
        cmap3 = plt.get_cmap("winter", 128)
        colors = np.vstack(
            (
                cmap1(np.linspace(0, 1, 128)),
                cmap2(np.linspace(0, 1, 128)),
                cmap3(np.linspace(0, 1, 128)),
            )
        )
        custom_cmap = mcolors.ListedColormap(colors, name="appended_colormap")
        cmap = plt.get_cmap(custom_cmap)
    else:
        val_min, val_max = min_max_vals
        cmap = plt.get_cmap("bwr_r", 128)

    norm = mpl.colors.Normalize(vmin=val_min, vmax=val_max, clip=True)
    val_mid = (val_max + val_min) / 2
    val_q3, val_q1 = (val_max + val_mid) / 2, (val_mid + val_min) / 2
    val_o1, val_o3, val_o5, val_o7 = (
        (val_q1 + val_min) / 2,
        (val_q1 + val_mid) / 2,
        (val_q3 + val_mid) / 2,
        (val_q3 + val_max) / 2,
    )
    colorfun = lambda x: np.rint(np.array(cmap(norm(x))) * 255)

    potential_surf_global = []
    for index, solute in enumerate(simulation.solutes):
        if solute.solute_name in solutes_plot or index in solutes_plot:
            if values == "phi":
                phi_vertices, unit_label = solute.get_surface_potential(
                    units=units, print_units=False
                )
            elif values == "d_phi":
                phi_vertices, unit_label = solute.get_surface_potential_derivative(
                    units=units,
                    print_units=False,
                    internal_derivative=internal_derivative_plot,
                )
            else:
                print("Unrecognized values, using phi as example")
                values = "phi"
                phi_vertices, unit_label = solute.get_surface_potential(
                    units=units, print_units=False
                )

            if location == "stern" and solute.stern_object:
                solute_mesh = solute.stern_mesh.mesh
            else:
                solute_mesh = solute.mesh

            surface_values = np.mean(phi_vertices[solute_mesh.elements.T], axis=1)
            color_codes = [
                "rgb({0}, {1}, {2})".format(*colorfun(x)) for x in surface_values
            ]
            color_codes = np.array(color_codes)

            potential_surf = ff.create_trisurf(
                x=solute_mesh.vertices[0, :],
                y=solute_mesh.vertices[1, :],
                z=solute_mesh.vertices[2, :],
                simplices=solute_mesh.elements.T,
                color_func=color_codes,
                plot_edges=False,
                showbackground=False,
                title=dict(text=f"{values} surface value"),
            )
            potential_surf_global.append(potential_surf)

    tick_vals = [
        val_min,
        val_o1,
        val_q1,
        val_o3,
        val_mid,
        val_o5,
        val_q3,
        val_o7,
        val_max,
    ]
    colorbar_codes = ["rgb({0}, {1}, {2})".format(*colorfun(v)) for v in tick_vals]

    plotly_colorscale = []
    for i, col in enumerate(colorbar_codes):
        plotly_colorscale.append([i / (len(colorbar_codes) - 1), col])

    colorbar_trace = go.Scatter3d(
        x=[0],
        y=[0],
        z=[0],
        mode="markers",
        scene="scene",
        opacity=0,
        hoverinfo="skip",
        marker=dict(
            color=[val_min],
            colorscale=plotly_colorscale,
            showscale=True,
            cmin=val_min,
            cmax=val_max,
            colorbar=dict(
                title=f"{unit_label}",
                thickness=15,
                tickvals=tick_vals,
                ticktext=["%1.2f" % v for v in tick_vals],
            ),
        ),
    )
    if isosurface_potential_vals[0] is not None:
        coordinates, potential, isosurface_potential = isosurface_potential_vals
        [X, Y, Z] = coordinates
        isosurface = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=potential.flatten(),
            opacity=0.3,
            isomin=isosurface_potential,
            isomax=isosurface_potential,
            colorscale="BlueRed",
            surface_count=1,  # number of isosurfaces, 2 by default: only min and max
            colorbar_nticks=1,  # colorbar ticks correspond to isosurface values
            showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False),
            surface_fill=0.4,
        )

    fig = go.Figure()

    for plot_surf in potential_surf_global:
        for trace in plot_surf.data:
            fig.add_trace(trace)

    fig.add_trace(colorbar_trace)
    if isosurface_potential_vals[0] is not None:
        fig.add_trace(isosurface)

    scene_config = dict(
        aspectmode="data",
        xaxis=dict(visible=show_axis, showbackground=False),
        yaxis=dict(visible=show_axis, showbackground=False),
        zaxis=dict(visible=show_axis, showbackground=False),
    )
    if camera_view:
        scene_config["camera"] = dict(
            eye=dict(x=camera_view[0], y=camera_view[1], z=camera_view[2])
        )

    fig.update_layout(scene=scene_config)
    fig.update_layout(width=figsize[0], height=figsize[1])

    plotly.offline.iplot(fig)
    if savefig:
        fig.write_image(name)

    return None
