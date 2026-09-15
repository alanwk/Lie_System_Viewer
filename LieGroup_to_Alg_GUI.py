from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update




"""
scratch
 -> 7 -> 12 -> 15 -> 11 ->
 w/ partner as 2
 commutator vec creates circle (gives deriv at each point with the direction)

"""


# ============================================================
# Configuration and group sampling
# ============================================================

IDENTITY_COLOR = "#2ca02c"
SELECTED_COLOR = "#FFCB05"       # Michigan maize
INVERSE_COLOR = "#00A6D6"
PARTNER_COLOR = "#4C78A8"
BRACKET_COLOR = "#D45087"
BACKGROUND_COLOR = "#F7F8FA"
DARK_COLOR = "#1F2937"

COMMUTATOR_FIELD_COLOR = "#8B5CF6"
COMMUTATOR_FIELD_SCALE = 0.16
COMMUTATOR_FIELD_OPACITY = 0.20
COMMUTATOR_FIELD_WIDTH = 4


def normalize_angle(theta: float) -> float:
    """Map an angle to [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


# def generate_group_data(
#     mode: str,
#     seed: int = 31415,
#     number_of_so3_samples: int = 180,
# ) -> dict:
#     """
#     Generate group elements represented by principal logarithm coordinates.

#     For SO(2), each element is represented by theta.

#     For SO(3), each element is represented by its principal rotation vector:
#         x = theta * axis
#     where ||x|| <= pi.
#     """
#     if mode == "SO2":
#         angles = np.linspace(-np.pi, np.pi, 120, endpoint=False)

#         # Remove the sampled zero so identity can be inserted exactly at index 0.
#         angles = angles[np.abs(angles) > 1e-10]
#         angles = np.concatenate(([0.0], angles))

#         return {
#             "mode": "SO2",
#             "coordinates": angles[:, None].tolist(),
#             "seed": seed,
#         }

#     rng = np.random.default_rng(seed)

#     # scipy's Rotation.random produces random rotations distributed over SO(3).
#     rotations = Rotation.random(
#         number_of_so3_samples - 1,
#         random_state=rng,
#     )

#     rotation_vectors = rotations.as_rotvec()

#     # Identity must always be index 0.
#     rotation_vectors = np.vstack(
#         [
#             np.zeros(3),
#             rotation_vectors,
#         ]
#     )

#     return {
#         "mode": "SO3",
#         "coordinates": rotation_vectors.tolist(),
#         "seed": seed,
#     }

def add_anchored_3d_vector(
    figure: go.Figure,
    origin: np.ndarray,
    vector: np.ndarray,
    name: str,
    color: str,
    width: int = 8,
    showlegend: bool = True,
) -> None:
    """
    Draw a 3D vector whose tail is attached to an arbitrary origin.

    Parameters
    ----------
    figure
        Plotly figure receiving the traces.
    origin
        Location of the vector's tail.
    vector
        Vector displacement. The endpoint is origin + vector.
    name
        Legend and hover label.
    color
        Plotly-compatible color.
    """
    origin = np.asarray(origin, dtype=float)
    vector = np.asarray(vector, dtype=float)

    magnitude = np.linalg.norm(vector)
    endpoint = origin + vector

    if magnitude < 1e-10:
        # A zero vector has no visible shaft, so mark its location explicitly.
        figure.add_trace(
            go.Scatter3d(
                x=[origin[0]],
                y=[origin[1]],
                z=[origin[2]],
                mode="markers+text",
                marker=dict(
                    size=12,
                    color=color,
                    symbol="diamond-open",
                    line=dict(
                        color=color,
                        width=4,
                    ),
                ),
                text=[f"{name} = 0"],
                textposition="middle right",
                name=name,
                showlegend=showlegend,
                hovertemplate=(
                    f"{name}<br>"
                    "Zero vector<br>"
                    f"Attached at "
                    f"({origin[0]:+.4f}, "
                    f"{origin[1]:+.4f}, "
                    f"{origin[2]:+.4f})"
                    "<extra></extra>"
                ),
            )
        )
        return

    # Vector shaft.
    figure.add_trace(
        go.Scatter3d(
            x=[origin[0], endpoint[0]],
            y=[origin[1], endpoint[1]],
            z=[origin[2], endpoint[2]],
            mode="lines",
            line=dict(
                color=color,
                width=width,
            ),
            name=name,
            showlegend=showlegend,
            hovertemplate=(
                f"{name}<br>"
                f"Tail = ({origin[0]:+.4f}, "
                f"{origin[1]:+.4f}, "
                f"{origin[2]:+.4f})<br>"
                f"Vector = ({vector[0]:+.4f}, "
                f"{vector[1]:+.4f}, "
                f"{vector[2]:+.4f})<br>"
                f"Tip = ({endpoint[0]:+.4f}, "
                f"{endpoint[1]:+.4f}, "
                f"{endpoint[2]:+.4f})"
                "<extra></extra>"
            ),
        )
    )

    # Arrowhead.
    unit_vector = vector / magnitude
    cone_length = min(0.22 * magnitude, 0.35)
    cone_start = endpoint - cone_length * unit_vector

    figure.add_trace(
        go.Cone(
            x=[cone_start[0]],
            y=[cone_start[1]],
            z=[cone_start[2]],
            u=[cone_length * unit_vector[0]],
            v=[cone_length * unit_vector[1]],
            w=[cone_length * unit_vector[2]],
            anchor="tail",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode="absolute",
            sizeref=max(0.10, cone_length),
            name=f"{name} arrowhead",
            showlegend=False,
            hoverinfo="skip",
        )
    )

def add_so3_commutator_vector_field(
    figure: go.Figure,
    coordinates: np.ndarray,
    fixed_element: np.ndarray,
    scale: float = COMMUTATOR_FIELD_SCALE,
    opacity: float = COMMUTATOR_FIELD_OPACITY,
    color: str = COMMUTATOR_FIELD_COLOR,
) -> np.ndarray:
    """
    Draw a downscaled commutator vector at every SO(3) rotation-vector
    coordinate.

    If Z is one displayed SO(3) coordinate and X is the selected element,
    the plotted vector is

        scale * [Z, X] = scale * (Z cross X).

    Thus, the selected element is held fixed while each grid point Z is
    treated as the first argument of the Lie bracket.

    The unscaled field satisfies

        dZ/dt = [Z, X],

    whose trajectories are adjoint-orbit circles about the axis defined
    by X.

    Parameters
    ----------
    figure
        Plotly figure receiving the traces.
    coordinates
        Array of shape (N, 3) containing principal rotation vectors.
    fixed_element
        Selected rotation vector X.
    scale
        Visual scale factor applied to every bracket vector.
    opacity
        Opacity of the field arrows.
    color
        Plotly-compatible color.

    Returns
    -------
    np.ndarray
        Endpoints of all scaled field vectors. These can be included when
        calculating the plot's axis limits.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    fixed_element = np.asarray(fixed_element, dtype=float)

    # V_X(Z) = [Z, X] = Z cross X.
    unscaled_vectors = np.cross(
        coordinates,
        fixed_element[None, :],
    )

    scaled_vectors = float(scale) * unscaled_vectors
    endpoints = coordinates + scaled_vectors

    magnitudes = np.linalg.norm(scaled_vectors, axis=1)
    nonzero_mask = magnitudes > 1e-10

    # If the selected element is the identity, X = 0 and therefore
    # [Z, X] = 0 at every point.
    if not np.any(nonzero_mask):
        figure.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(
                    size=1,
                    color=color,
                    opacity=0,
                ),
                name="[Z,X] field = 0 because X = 0",
                visible="legendonly",
                hoverinfo="skip",
            )
        )

        return endpoints

    origins = coordinates[nonzero_mask]
    vectors = scaled_vectors[nonzero_mask]
    tips = endpoints[nonzero_mask]
    displayed_magnitudes = magnitudes[nonzero_mask]
    original_magnitudes = np.linalg.norm(
        unscaled_vectors[nonzero_mask],
        axis=1,
    )

    # --------------------------------------------------------
    # Draw all vector shafts in a single Scatter3d trace.
    # None separates the individual line segments.
    # --------------------------------------------------------
    line_x = []
    line_y = []
    line_z = []

    for origin, tip in zip(origins, tips):
        line_x.extend([origin[0], tip[0], None])
        line_y.extend([origin[1], tip[1], None])
        line_z.extend([origin[2], tip[2], None])

    figure.add_trace(
        go.Scatter3d(
            x=line_x,
            y=line_y,
            z=line_z,
            mode="lines",
            line=dict(
                color=color,
                width=COMMUTATOR_FIELD_WIDTH,
            ),
            opacity=opacity,
            name=f"Field: {scale:.2f}[Z,X]",
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # Add small arrowheads using one Cone trace.
    #
    # Arrowhead length is proportional to the displayed vector
    # for short vectors and capped for long vectors.
    # --------------------------------------------------------
    unit_vectors = vectors / displayed_magnitudes[:, None]

    arrowhead_lengths = np.minimum(
        0.25 * displayed_magnitudes,
        0.16,
    )

    arrowhead_vectors = (
        arrowhead_lengths[:, None] * unit_vectors
    )

    arrowhead_origins = tips - arrowhead_vectors

    figure.add_trace(
        go.Cone(
            x=arrowhead_origins[:, 0],
            y=arrowhead_origins[:, 1],
            z=arrowhead_origins[:, 2],
            u=arrowhead_vectors[:, 0],
            v=arrowhead_vectors[:, 1],
            w=arrowhead_vectors[:, 2],
            anchor="tail",
            colorscale=[
                [0, color],
                [1, color],
            ],
            showscale=False,
            sizemode="absolute",
            sizeref=0.10,
            opacity=opacity,
            name="Commutator-field arrowheads",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # --------------------------------------------------------
    # Optional nearly invisible points supplying useful hover
    # information at the vector tails.
    # --------------------------------------------------------
    hover_text = [
        (
            f"Coordinate Z = "
            f"({origin[0]:+.4f}, "
            f"{origin[1]:+.4f}, "
            f"{origin[2]:+.4f})"
            f"<br>[Z,X] = "
            f"({unscaled[0]:+.4f}, "
            f"{unscaled[1]:+.4f}, "
            f"{unscaled[2]:+.4f})"
            f"<br>||[Z,X]|| = {original_norm:.4f}"
            f"<br>Displayed scale = {scale:.3f}"
        )
        for origin, unscaled, original_norm in zip(
            origins,
            unscaled_vectors[nonzero_mask],
            original_magnitudes,
        )
    ]

    figure.add_trace(
        go.Scatter3d(
            x=origins[:, 0],
            y=origins[:, 1],
            z=origins[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=color,
                opacity=0.01,
            ),
            text=hover_text,
            name="Commutator-field values",
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    return endpoints

def generate_so3_grid(target_count: int = 100) -> dict:
    """
    Generate an approximately target-sized Cartesian grid inside the
    principal axis-angle ball for SO(3).

    All neighboring lattice points have the same Euclidean spacing in
    rotation-vector coordinates.

    Notes
    -----
    This grid is equidistant in the displayed axis-angle coordinate chart,
    not with respect to the global geodesic metric on SO(3).

    Complete integer-radius shells are retained so that the grid remains
    symmetric about the identity.
    """
    target_count = int(target_count)

    # Large enough for all requested target sizes.
    maximum_integer_coordinate = 10

    integer_values = np.arange(
        -maximum_integer_coordinate,
        maximum_integer_coordinate + 1,
        dtype=int,
    )

    grid_x, grid_y, grid_z = np.meshgrid(
        integer_values,
        integer_values,
        integer_values,
        indexing="ij",
    )

    integer_grid = np.column_stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
            grid_z.ravel(),
        ]
    )

    squared_radii = np.sum(integer_grid**2, axis=1)

    # Determine which complete lattice shell gives the closest point count.
    possible_squared_cutoffs = np.unique(squared_radii)
    possible_squared_cutoffs = possible_squared_cutoffs[
        possible_squared_cutoffs > 0
    ]

    best_squared_cutoff = min(
        possible_squared_cutoffs,
        key=lambda cutoff: (
            abs(np.count_nonzero(squared_radii <= cutoff) - target_count),
            cutoff,
        ),
    )

    keep_mask = squared_radii <= best_squared_cutoff
    retained_integer_grid = integer_grid[keep_mask]
    retained_squared_radii = squared_radii[keep_mask]

    # Keep the outermost shell just inside pi. This avoids placing samples
    # exactly on the axis-angle branch boundary, where antipodal points
    # represent the same rotation.
    maximum_rotation_angle = np.pi * (1.0 - 1e-6)

    grid_spacing = (
        maximum_rotation_angle / np.sqrt(float(best_squared_cutoff))
    )

    rotation_vectors = retained_integer_grid.astype(float) * grid_spacing

    # Put the identity first, as required by the rest of the application.
    identity_mask = retained_squared_radii == 0
    identity = rotation_vectors[identity_mask]

    nonidentity_vectors = rotation_vectors[~identity_mask]
    nonidentity_squared_radii = retained_squared_radii[~identity_mask]

    # Sort by distance from identity, followed by coordinates. This makes
    # element numbering stable and reasonably intuitive.
    sorting_order = np.lexsort(
        (
            nonidentity_vectors[:, 2],
            nonidentity_vectors[:, 1],
            nonidentity_vectors[:, 0],
            nonidentity_squared_radii,
        )
    )

    rotation_vectors = np.vstack(
        [
            identity,
            nonidentity_vectors[sorting_order],
        ]
    )

    return {
        "mode": "SO3",
        "coordinates": rotation_vectors.tolist(),
        "requested_count": target_count,
        "actual_count": len(rotation_vectors),
        "grid_spacing": float(grid_spacing),
        "squared_shell_cutoff": int(best_squared_cutoff),
    }


def generate_group_data(
    mode: str,
    number_of_so3_samples: int = 100,
) -> dict:
    """
    Generate SO(2) samples or an approximately requested-size SO(3) grid.
    """
    if mode == "SO2":
        angles = np.linspace(
            -np.pi,
            np.pi,
            120,
            endpoint=False,
        )

        # Remove the sampled zero so identity can be inserted exactly once
        # at index zero.
        angles = angles[np.abs(angles) > 1e-10]
        angles = np.concatenate(([0.0], angles))

        return {
            "mode": "SO2",
            "coordinates": angles[:, None].tolist(),
            "requested_count": len(angles),
            "actual_count": len(angles),
        }

    return generate_so3_grid(
        target_count=number_of_so3_samples,
    )


def element_label(mode: str, index: int, coordinate: np.ndarray) -> str:
    if index == 0:
        return "0 — Identity"

    if mode == "SO2":
        theta = coordinate[0]
        return f"{index} — θ = {theta:+.3f} rad"

    angle = np.linalg.norm(coordinate)
    if angle < 1e-12:
        return f"{index} — Identity"

    axis = coordinate / angle
    return (
        f"{index} — angle {angle:.3f}, "
        f"axis ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f})"
    )


def make_dropdown_options(data: dict) -> list[dict]:
    mode = data["mode"]
    coordinates = np.asarray(data["coordinates"], dtype=float)

    return [
        {
            "label": element_label(mode, i, coordinate),
            "value": i,
        }
        for i, coordinate in enumerate(coordinates)
    ]


INITIAL_DATA = generate_group_data("SO2")
INITIAL_OPTIONS = make_dropdown_options(INITIAL_DATA)


# ============================================================
# Mathematical operations
# ============================================================

def rotation_matrix_so2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c, -s],
            [s, c],
        ]
    )


def selected_group_information(
    data: dict,
    selected_index: int,
    partner_index: int,
) -> dict:
    mode = data["mode"]
    coordinates = np.asarray(data["coordinates"], dtype=float)

    x = coordinates[selected_index]
    y = coordinates[partner_index]

    if mode == "SO2":
        theta_x = float(x[0])
        theta_y = float(y[0])

        matrix = rotation_matrix_so2(theta_x)
        inverse_coordinate = np.array([-normalize_angle(theta_x)])

        # SO(2) is Abelian, so its Lie algebra bracket is always zero.
        bracket = np.array([0.0])

        return {
            "x": x,
            "y": y,
            "bracket": bracket,
            "matrix": matrix,
            "inverse_coordinate": inverse_coordinate,
        }

    rotation = Rotation.from_rotvec(x)
    matrix = rotation.as_matrix()

    # Under the standard R^3 identification of so(3):
    # [x, y] = x cross y
    bracket = np.cross(x, y)

    # exp(x)^(-1) = exp(-x).
    inverse_coordinate = -x

    return {
        "x": x,
        "y": y,
        "bracket": bracket,
        "matrix": matrix,
        "inverse_coordinate": inverse_coordinate,
    }


# ============================================================
# General plotting helpers
# ============================================================

def blank_figure(title: str) -> go.Figure:
    figure = go.Figure()

    figure.update_layout(
        title=title,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
        ),
    )

    return figure


def add_3d_vector(
    figure: go.Figure,
    vector: np.ndarray,
    name: str,
    color: str,
    width: int = 7,
    showlegend: bool = True,
) -> None:
    """Add a vector from the origin using a line and cone."""
    vector = np.asarray(vector, dtype=float)
    magnitude = np.linalg.norm(vector)

    if magnitude < 1e-10:
        figure.add_trace(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=color,
                    symbol="diamond",
                    line=dict(color="black", width=1),
                ),
                text=[f"{name} = 0"],
                textposition="top center",
                name=name,
                showlegend=showlegend,
                hovertemplate=f"{name} = 0<extra></extra>",
            )
        )
        return

    figure.add_trace(
        go.Scatter3d(
            x=[0, vector[0]],
            y=[0, vector[1]],
            z=[0, vector[2]],
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            showlegend=showlegend,
            hovertemplate=(
                f"{name}<br>"
                f"x = {vector[0]:+.4f}<br>"
                f"y = {vector[1]:+.4f}<br>"
                f"z = {vector[2]:+.4f}"
                "<extra></extra>"
            ),
        )
    )

    # A cone supplies the arrowhead.
    unit = vector / magnitude
    cone_length = min(0.22 * magnitude, 0.25)

    cone_start = vector - cone_length * unit

    figure.add_trace(
        go.Cone(
            x=[cone_start[0]],
            y=[cone_start[1]],
            z=[cone_start[2]],
            u=[cone_length * unit[0]],
            v=[cone_length * unit[1]],
            w=[cone_length * unit[2]],
            anchor="tail",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode="absolute",
            sizeref=max(0.10, cone_length),
            name=f"{name} arrowhead",
            showlegend=False,
            hoverinfo="skip",
        )
    )


def add_axis_line(
    figure: go.Figure,
    endpoint: np.ndarray,
    color: str,
    label: str,
) -> None:
    endpoint = np.asarray(endpoint, dtype=float)

    figure.add_trace(
        go.Scatter3d(
            x=[0, endpoint[0]],
            y=[0, endpoint[1]],
            z=[0, endpoint[2]],
            mode="lines+text",
            line=dict(color=color, width=7),
            text=["", label],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )


# ============================================================
# Physical-space plots
# ============================================================

def make_so2_physical_figure(theta: float) -> go.Figure:
    figure = blank_figure("A. Physical space: SO(2) acting on ℝ²")

    # An asymmetric arrow-like object makes orientation visually apparent.
    shape = np.array(
        [
            [0.00, 0.00],
            [1.35, 0.00],
            [1.05, 0.26],
            [1.05, 0.12],
            [0.25, 0.12],
            [0.25, 0.42],
            [0.00, 0.42],
            [0.00, 0.00],
        ]
    )

    rotation = rotation_matrix_so2(theta)
    transformed_shape = (rotation @ shape.T).T

    figure.add_trace(
        go.Scatter3d(
            x=shape[:, 0],
            y=shape[:, 1],
            z=np.zeros(len(shape)),
            mode="lines",
            line=dict(color="#9CA3AF", width=7, dash="dash"),
            name="Original object",
            hoverinfo="skip",
        )
    )

    figure.add_trace(
        go.Scatter3d(
            x=transformed_shape[:, 0],
            y=transformed_shape[:, 1],
            z=np.zeros(len(shape)),
            mode="lines+markers",
            line=dict(color=SELECTED_COLOR, width=9),
            marker=dict(size=3, color=DARK_COLOR),
            name="Selected transformation",
            hovertemplate=(
                "Transformed point<br>"
                "x = %{x:.3f}<br>"
                "y = %{y:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # Reference grid.
    for grid_value in np.linspace(-1.5, 1.5, 7):
        figure.add_trace(
            go.Scatter3d(
                x=[-1.5, 1.5],
                y=[grid_value, grid_value],
                z=[-0.01, -0.01],
                mode="lines",
                line=dict(color="#E5E7EB", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[grid_value, grid_value],
                y=[-1.5, 1.5],
                z=[-0.01, -0.01],
                mode="lines",
                line=dict(color="#E5E7EB", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        scene=dict(
            xaxis=dict(title="x", range=[-1.6, 1.6]),
            yaxis=dict(title="y", range=[-1.6, 1.6]),
            zaxis=dict(title="", range=[-0.2, 0.2], visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=0, y=0, z=2.6)),
        )
    )

    return figure


def cube_geometry() -> tuple[np.ndarray, list[tuple[int, int]]]:
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    ) * 0.65

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return vertices, edges


def add_cube(
    figure: go.Figure,
    vertices: np.ndarray,
    edges: list[tuple[int, int]],
    color: str,
    name: str,
    dash: str | None = None,
) -> None:
    first_edge = True

    for start, end in edges:
        points = vertices[[start, end]]

        figure.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="lines",
                line=dict(color=color, width=6, dash=dash),
                name=name,
                legendgroup=name,
                showlegend=first_edge,
                hoverinfo="skip",
            )
        )
        first_edge = False


def make_so3_physical_figure(rotation_vector: np.ndarray) -> go.Figure:
    figure = blank_figure("A. Physical space: SO(3) acting on ℝ³")

    rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
    vertices, edges = cube_geometry()
    transformed_vertices = (rotation @ vertices.T).T

    add_cube(
        figure,
        vertices,
        edges,
        color="#AAB2BD",
        name="Original cube",
        dash="dash",
    )

    add_cube(
        figure,
        transformed_vertices,
        edges,
        color=SELECTED_COLOR,
        name="Transformed cube",
    )

    # Original reference axes.
    add_axis_line(figure, np.array([1.3, 0, 0]), "#EF4444", "x")
    add_axis_line(figure, np.array([0, 1.3, 0]), "#22C55E", "y")
    add_axis_line(figure, np.array([0, 0, 1.3]), "#3B82F6", "z")

    # Rotated body axes.
    body_axis_colors = ["#991B1B", "#166534", "#1E40AF"]
    body_axis_labels = ["R e₁", "R e₂", "R e₃"]

    for i in range(3):
        add_3d_vector(
            figure,
            1.15 * rotation[:, i],
            body_axis_labels[i],
            body_axis_colors[i],
            width=5,
        )

    figure.update_layout(
        scene=dict(
            xaxis=dict(title="x", range=[-1.6, 1.6]),
            yaxis=dict(title="y", range=[-1.6, 1.6]),
            zaxis=dict(title="z", range=[-1.6, 1.6]),
            aspectmode="cube",
        )
    )

    return figure


# ============================================================
# Group-manifold plots
# ============================================================

# def make_so2_manifold_figure(
#     coordinates: np.ndarray,
#     selected_index: int,
#     inverse_coordinate: np.ndarray,
# ) -> go.Figure:
#     figure = blank_figure("B. Group manifold: SO(2) ≅ a circle")

#     angles = coordinates[:, 0]
#     x = np.cos(angles)
#     y = np.sin(angles)

#     figure.add_trace(
#         go.Scatter3d(
#             x=x,
#             y=y,
#             z=np.zeros_like(x),
#             mode="lines+markers",
#             line=dict(color="#9CA3AF", width=4),
#             marker=dict(size=4, color="#6B7280"),
#             customdata=np.arange(len(angles)),
#             name="SO(2) elements",
#             hovertemplate=(
#                 "Element %{customdata}<br>"
#                 "cos θ = %{x:.3f}<br>"
#                 "sin θ = %{y:.3f}<br>"
#                 "Click to select"
#                 "<extra></extra>"
#             ),
#         )
#     )

#     # Identity marker.
#     figure.add_trace(
#         go.Scatter3d(
#             x=[1],
#             y=[0],
#             z=[0],
#             mode="markers+text",
#             marker=dict(
#                 size=10,
#                 color=IDENTITY_COLOR,
#                 symbol="diamond",
#                 line=dict(color="black", width=1),
#             ),
#             text=["Identity"],
#             textposition="bottom center",
#             name="Identity",
#             hovertemplate="Identity rotation<extra></extra>",
#         )
#     )

#     selected_theta = coordinates[selected_index, 0]

#     figure.add_trace(
#         go.Scatter3d(
#             x=[np.cos(selected_theta)],
#             y=[np.sin(selected_theta)],
#             z=[0],
#             mode="markers+text",
#             marker=dict(
#                 size=13,
#                 color=SELECTED_COLOR,
#                 symbol="circle",
#                 line=dict(color="black", width=2),
#             ),
#             text=["Selected"],
#             textposition="top center",
#             name="Selected element",
#             hovertemplate=f"Selected θ = {selected_theta:+.4f}<extra></extra>",
#         )
#     )

#     if abs(selected_theta) > 1e-10:
#         inverse_theta = inverse_coordinate[0]

#         figure.add_trace(
#             go.Scatter3d(
#                 x=[np.cos(inverse_theta)],
#                 y=[np.sin(inverse_theta)],
#                 z=[0],
#                 mode="markers+text",
#                 marker=dict(
#                     size=12,
#                     color=INVERSE_COLOR,
#                     symbol="x",
#                     line=dict(color="black", width=2),
#                 ),
#                 text=["Inverse"],
#                 textposition="bottom center",
#                 name="Inverse element",
#                 hovertemplate=(
#                     f"Inverse rotation θ = {inverse_theta:+.4f}"
#                     "<extra></extra>"
#                 ),
#             )
#         )

#     figure.update_layout(
#         scene=dict(
#             xaxis=dict(title="cos θ", range=[-1.25, 1.25]),
#             yaxis=dict(title="sin θ", range=[-1.25, 1.25]),
#             zaxis=dict(visible=False, range=[-0.2, 0.2]),
#             aspectmode="cube",
#             camera=dict(eye=dict(x=0, y=0, z=2.5)),
#         )
#     )

#     return figure

def make_so2_manifold_figure(
    coordinates: np.ndarray,
    selected_index: int,
    partner_index: int,
    inverse_coordinate: np.ndarray,
    bracket: np.ndarray,
) -> go.Figure:
    figure = blank_figure(
        "B. Group manifold: SO(2) ≅ a circle"
    )

    angles = coordinates[:, 0]
    circle_x = np.cos(angles)
    circle_y = np.sin(angles)

    figure.add_trace(
        go.Scatter3d(
            x=circle_x,
            y=circle_y,
            z=np.zeros_like(circle_x),
            mode="lines+markers",
            line=dict(
                color="#9CA3AF",
                width=4,
            ),
            marker=dict(
                size=4,
                color="#6B7280",
            ),
            customdata=np.arange(len(angles)),
            name="SO(2) elements",
            hovertemplate=(
                "Element %{customdata}<br>"
                "cos θ = %{x:.3f}<br>"
                "sin θ = %{y:.3f}<br>"
                "Click to select"
                "<extra></extra>"
            ),
        )
    )

    # Identity.
    figure.add_trace(
        go.Scatter3d(
            x=[1],
            y=[0],
            z=[0],
            mode="markers+text",
            marker=dict(
                size=10,
                color=IDENTITY_COLOR,
                symbol="diamond",
                line=dict(
                    color="black",
                    width=1,
                ),
            ),
            text=["Identity"],
            textposition="bottom center",
            name="Identity",
            hovertemplate="Identity rotation<extra></extra>",
        )
    )

    selected_theta = float(coordinates[selected_index, 0])
    partner_theta = float(coordinates[partner_index, 0])

    selected_point = np.array(
        [
            np.cos(selected_theta),
            np.sin(selected_theta),
            0.0,
        ]
    )

    partner_point = np.array(
        [
            np.cos(partner_theta),
            np.sin(partner_theta),
            0.0,
        ]
    )

    # Partner h marker. Add this before the selected marker so that the
    # selected marker stays visually dominant if g == h.
    figure.add_trace(
        go.Scatter3d(
            x=[partner_point[0]],
            y=[partner_point[1]],
            z=[partner_point[2]],
            mode="markers+text",
            marker=dict(
                size=12,
                color=PARTNER_COLOR,
                symbol="square",
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
            text=["Partner h"],
            textposition="middle left",
            name="Partner element h",
            hovertemplate=(
                f"Partner h<br>"
                f"Index = {partner_index}<br>"
                f"θ = {partner_theta:+.4f}"
                "<extra></extra>"
            ),
        )
    )

    # Selected g marker.
    figure.add_trace(
        go.Scatter3d(
            x=[selected_point[0]],
            y=[selected_point[1]],
            z=[selected_point[2]],
            mode="markers+text",
            marker=dict(
                size=13,
                color=SELECTED_COLOR,
                symbol="circle",
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
            text=["Selected g"],
            textposition="top center",
            name="Selected element g",
            hovertemplate=(
                f"Selected g<br>"
                f"Index = {selected_index}<br>"
                f"θ = {selected_theta:+.4f}"
                "<extra></extra>"
            ),
        )
    )

    # Inverse marker.
    if abs(selected_theta) > 1e-10:
        inverse_theta = float(inverse_coordinate[0])

        figure.add_trace(
            go.Scatter3d(
                x=[np.cos(inverse_theta)],
                y=[np.sin(inverse_theta)],
                z=[0],
                mode="markers+text",
                marker=dict(
                    size=12,
                    color=INVERSE_COLOR,
                    symbol="x",
                    line=dict(
                        color="black",
                        width=2,
                    ),
                ),
                text=["g⁻¹"],
                textposition="bottom center",
                name="Inverse element g⁻¹",
                hovertemplate=(
                    f"Inverse of g<br>"
                    f"θ = {inverse_theta:+.4f}"
                    "<extra></extra>"
                ),
            )
        )

    # A scalar Lie-algebra vector in so(2) maps to a tangent vector on the
    # embedded circle through d/dθ (cos θ, sin θ).
    circle_tangent_direction = np.array(
        [
            -np.sin(selected_theta),
            np.cos(selected_theta),
            0.0,
        ]
    )

    embedded_bracket = float(bracket[0]) * circle_tangent_direction

    add_anchored_3d_vector(
        figure=figure,
        origin=selected_point,
        vector=embedded_bracket,
        name="[X,Y] attached at g",
        color=BRACKET_COLOR,
        width=9,
    )

    figure.add_annotation(
        text=(
            "The bracket vector is attached to the selected element g.<br>"
            "For SO(2), [X,Y] = 0 because rotations commute."
        ),
        x=0.5,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
    )

    figure.update_layout(
        scene=dict(
            xaxis=dict(
                title="cos θ",
                range=[-1.45, 1.45],
            ),
            yaxis=dict(
                title="sin θ",
                range=[-1.45, 1.45],
            ),
            zaxis=dict(
                visible=False,
                range=[-0.2, 0.2],
            ),
            aspectmode="cube",
            camera=dict(
                eye=dict(
                    x=0,
                    y=0,
                    z=2.5,
                )
            ),
        )
    )

    return figure


# def make_so3_manifold_figure(
#     coordinates: np.ndarray,
#     selected_index: int,
#     inverse_coordinate: np.ndarray,
# ) -> go.Figure:
#     figure = blank_figure(
#         "B. SO(3) grid in principal axis-angle coordinates"
#     )

#     norms = np.linalg.norm(coordinates, axis=1)

#     figure.add_trace(
#         go.Scatter3d(
#             x=coordinates[:, 0],
#             y=coordinates[:, 1],
#             z=coordinates[:, 2],
#             mode="markers",
#             marker=dict(
#                 size=4,
#                 color=norms,
#                 colorscale="Viridis",
#                 cmin=0,
#                 cmax=np.pi,
#                 opacity=0.70,
#                 colorbar=dict(
#                     title="Rotation<br>angle",
#                     thickness=12,
#                 ),
#             ),
#             customdata=np.arange(len(coordinates)),
#             name="SO(3) coordinate-grid elements",
#             hovertemplate=(
#                 "Element %{customdata}<br>"
#                 "r₁ = %{x:.3f}<br>"
#                 "r₂ = %{y:.3f}<br>"
#                 "r₃ = %{z:.3f}<br>"
#                 "Click to select"
#                 "<extra></extra>"
#             ),
#         )
#     )

#     # Draw the radius-pi boundary of the principal axis-angle ball.
#     u = np.linspace(0, 2 * np.pi, 45)
#     v = np.linspace(0, np.pi, 25)
#     radius = np.pi

#     sphere_x = radius * np.outer(np.cos(u), np.sin(v))
#     sphere_y = radius * np.outer(np.sin(u), np.sin(v))
#     sphere_z = radius * np.outer(np.ones_like(u), np.cos(v))

#     figure.add_trace(
#         go.Surface(
#             x=sphere_x,
#             y=sphere_y,
#             z=sphere_z,
#             opacity=0.07,
#             showscale=False,
#             colorscale=[[0, "#6B7280"], [1, "#6B7280"]],
#             name="Angle π boundary",
#             hoverinfo="skip",
#             showlegend=False,
#         )
#     )

#     figure.add_trace(
#         go.Scatter3d(
#             x=[0],
#             y=[0],
#             z=[0],
#             mode="markers+text",
#             marker=dict(
#                 size=10,
#                 color=IDENTITY_COLOR,
#                 symbol="diamond",
#                 line=dict(color="black", width=1),
#             ),
#             text=["Identity"],
#             textposition="bottom center",
#             name="Identity",
#         )
#     )

#     selected = coordinates[selected_index]

#     figure.add_trace(
#         go.Scatter3d(
#             x=[selected[0]],
#             y=[selected[1]],
#             z=[selected[2]],
#             mode="markers+text",
#             marker=dict(
#                 size=13,
#                 color=SELECTED_COLOR,
#                 symbol="circle",
#                 line=dict(color="black", width=2),
#             ),
#             text=["Selected"],
#             textposition="top center",
#             name="Selected element",
#             hovertemplate=(
#                 f"Selected rotvec<br>"
#                 f"({selected[0]:+.4f}, "
#                 f"{selected[1]:+.4f}, "
#                 f"{selected[2]:+.4f})"
#                 "<extra></extra>"
#             ),
#         )
#     )

#     if np.linalg.norm(selected) > 1e-10:
#         figure.add_trace(
#             go.Scatter3d(
#                 x=[inverse_coordinate[0]],
#                 y=[inverse_coordinate[1]],
#                 z=[inverse_coordinate[2]],
#                 mode="markers+text",
#                 marker=dict(
#                     size=12,
#                     color=INVERSE_COLOR,
#                     symbol="x",
#                     line=dict(color="black", width=2),
#                 ),
#                 text=["Inverse"],
#                 textposition="bottom center",
#                 name="Inverse element",
#                 hovertemplate=(
#                     "Inverse rotvec<br>"
#                     f"({inverse_coordinate[0]:+.4f}, "
#                     f"{inverse_coordinate[1]:+.4f}, "
#                     f"{inverse_coordinate[2]:+.4f})"
#                     "<extra></extra>"
#                 ),
#             )
#         )

#     figure.update_layout(
#         scene=dict(
#             xaxis=dict(title="r₁", range=[-3.4, 3.4]),
#             yaxis=dict(title="r₂", range=[-3.4, 3.4]),
#             zaxis=dict(title="r₃", range=[-3.4, 3.4]),
#             aspectmode="cube",
#         )
#     )

#     return figure

def make_so3_manifold_figure(
    coordinates: np.ndarray,
    selected_index: int,
    partner_index: int,
    inverse_coordinate: np.ndarray,
    bracket: np.ndarray,
    show_commutator_field: bool = True,
) -> go.Figure:
    figure = blank_figure(
        "B. SO(3) grid in principal axis-angle coordinates"
    )

    norms = np.linalg.norm(coordinates, axis=1)

    selected = np.asarray(
        coordinates[selected_index],
        dtype=float,
    )

    partner = np.asarray(
        coordinates[partner_index],
        dtype=float,
    )

    figure.add_trace(
        go.Scatter3d(
            x=coordinates[:, 0],
            y=coordinates[:, 1],
            z=coordinates[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=norms,
                colorscale="Viridis",
                cmin=0,
                cmax=np.pi,
                opacity=0.70,
                colorbar=dict(
                    title="Rotation<br>angle",
                    thickness=12,
                ),
            ),
            customdata=np.arange(len(coordinates)),
            name="SO(3) coordinate-grid elements",
            hovertemplate=(
                "Element %{customdata}<br>"
                "r₁ = %{x:.3f}<br>"
                "r₂ = %{y:.3f}<br>"
                "r₃ = %{z:.3f}<br>"
                "Click to select"
                "<extra></extra>"
            ),
        )
    )

    # Radius-pi boundary of the principal axis-angle ball.
    u = np.linspace(0, 2 * np.pi, 45)
    v = np.linspace(0, np.pi, 25)
    radius = np.pi

    sphere_x = radius * np.outer(
        np.cos(u),
        np.sin(v),
    )
    sphere_y = radius * np.outer(
        np.sin(u),
        np.sin(v),
    )
    sphere_z = radius * np.outer(
        np.ones_like(u),
        np.cos(v),
    )

    figure.add_trace(
        go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            opacity=0.07,
            showscale=False,
            colorscale=[
                [0, "#6B7280"],
                [1, "#6B7280"],
            ],
            name="Angle π boundary",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # At every coordinate Z, show a transparent and downscaled version of
    #
    #     [Z, X] = Z cross X,
    #
    # where X is the currently selected element. These vectors are tangent
    # to the adjoint-orbit flow generated by X.
    if show_commutator_field:
        field_endpoints = add_so3_commutator_vector_field(
            figure=figure,
            coordinates=coordinates,
            fixed_element=selected,
            scale=COMMUTATOR_FIELD_SCALE,
            opacity=COMMUTATOR_FIELD_OPACITY,
            color=COMMUTATOR_FIELD_COLOR,
        )
    else:
        field_endpoints = coordinates.copy()

    # Identity marker.
    figure.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers+text",
            marker=dict(
                size=10,
                color=IDENTITY_COLOR,
                symbol="diamond",
                line=dict(
                    color="black",
                    width=1,
                ),
            ),
            text=["Identity"],
            textposition="bottom center",
            name="Identity",
            hovertemplate="Identity rotation<extra></extra>",
        )
    )

    # Partner h. Draw it before g so g remains the visually dominant marker
    # when the two indices are equal.
    figure.add_trace(
        go.Scatter3d(
            x=[partner[0]],
            y=[partner[1]],
            z=[partner[2]],
            mode="markers+text",
            marker=dict(
                size=12,
                color=PARTNER_COLOR,
                symbol="square",
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
            text=["Partner h"],
            textposition="middle left",
            name="Partner element h",
            hovertemplate=(
                f"Partner h<br>"
                f"Index = {partner_index}<br>"
                f"rotvec = ({partner[0]:+.4f}, "
                f"{partner[1]:+.4f}, "
                f"{partner[2]:+.4f})"
                "<extra></extra>"
            ),
        )
    )

    # Selected g.
    figure.add_trace(
        go.Scatter3d(
            x=[selected[0]],
            y=[selected[1]],
            z=[selected[2]],
            mode="markers+text",
            marker=dict(
                size=13,
                color=SELECTED_COLOR,
                symbol="circle",
                line=dict(
                    color="black",
                    width=2,
                ),
            ),
            text=["Selected g"],
            textposition="top center",
            name="Selected element g",
            hovertemplate=(
                f"Selected g<br>"
                f"Index = {selected_index}<br>"
                f"rotvec = ({selected[0]:+.4f}, "
                f"{selected[1]:+.4f}, "
                f"{selected[2]:+.4f})"
                "<extra></extra>"
            ),
        )
    )

    # Inverse of g.
    if np.linalg.norm(selected) > 1e-10:
        figure.add_trace(
            go.Scatter3d(
                x=[inverse_coordinate[0]],
                y=[inverse_coordinate[1]],
                z=[inverse_coordinate[2]],
                mode="markers+text",
                marker=dict(
                    size=12,
                    color=INVERSE_COLOR,
                    symbol="x",
                    line=dict(
                        color="black",
                        width=2,
                    ),
                ),
                text=["g⁻¹"],
                textposition="bottom center",
                name="Inverse element g⁻¹",
                hovertemplate=(
                    "Inverse of g<br>"
                    f"rotvec = ({inverse_coordinate[0]:+.4f}, "
                    f"{inverse_coordinate[1]:+.4f}, "
                    f"{inverse_coordinate[2]:+.4f})"
                    "<extra></extra>"
                ),
            )
        )

    # Attach the Lie-bracket vector to the selected group element in the
    # displayed axis-angle chart.
    add_anchored_3d_vector(
        figure=figure,
        origin=selected,
        vector=bracket,
        name="[X,Y] attached at g",
        color=BRACKET_COLOR,
        width=9,
    )

    bracket_endpoint = selected + bracket

    figure.add_annotation(
        text=(
            "X = log(g), Y = log(h), and [X,Y] = X × Y.<br>"
            "The magenta vector is translated so its tail is attached to g."
        ),
        x=0.5,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
    )

    # The bracket endpoint can fall outside the radius-pi coordinate ball.
    # Expand the axes enough to keep the complete vector visible.
    displayed_points = np.vstack(
        [
            coordinates,
            field_endpoints,
            selected[None, :],
            partner[None, :],
            inverse_coordinate[None, :],
            bracket_endpoint[None, :],
        ]
    )

    axis_limit = max(
        np.pi * 1.10,
        float(np.max(np.abs(displayed_points))) * 1.12,
    )

    figure.update_layout(
        scene=dict(
            xaxis=dict(
                title="r₁",
                range=[-axis_limit, axis_limit],
            ),
            yaxis=dict(
                title="r₂",
                range=[-axis_limit, axis_limit],
            ),
            zaxis=dict(
                title="r₃",
                range=[-axis_limit, axis_limit],
            ),
            aspectmode="cube",
        )
    )

    return figure


# ============================================================
# Lie-algebra plots
# ============================================================

def make_so2_algebra_figure(
    x: np.ndarray,
    y: np.ndarray,
    bracket: np.ndarray,
) -> go.Figure:
    figure = blank_figure("C. Lie algebra so(2): a one-dimensional line")

    figure.add_trace(
        go.Scatter3d(
            x=[-np.pi, np.pi],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            line=dict(color="#9CA3AF", width=4),
            name="so(2)",
            hoverinfo="skip",
        )
    )

    add_3d_vector(
        figure,
        np.array([x[0], 0, 0]),
        "X = log(selected)",
        SELECTED_COLOR,
    )

    add_3d_vector(
        figure,
        np.array([y[0], 0, 0]),
        "Y = log(partner)",
        PARTNER_COLOR,
    )

    # The bracket is always zero for so(2).
    add_3d_vector(
        figure,
        np.array([bracket[0], 0, 0]),
        "[X,Y] = 0",
        BRACKET_COLOR,
    )

    figure.add_annotation(
        text=(
            "SO(2) is Abelian, so [X,Y] = 0 for every X and Y.<br>"
            "The logarithm has a branch cut at ±π."
        ),
        x=0.5,
        y=0.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
    )

    figure.update_layout(
        scene=dict(
            xaxis=dict(title="Angular velocity ω", range=[-3.5, 3.5]),
            yaxis=dict(title="", range=[-0.6, 0.6], visible=False),
            zaxis=dict(title="", range=[-0.6, 0.6], visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=0, y=-2.0, z=1.4)),
        )
    )

    return figure


def make_so3_algebra_figure(
    x: np.ndarray,
    y: np.ndarray,
    bracket: np.ndarray,
) -> go.Figure:
    figure = blank_figure("C. Lie algebra so(3) ≅ ℝ³")

    add_3d_vector(
        figure,
        x,
        "X = log(selected)",
        SELECTED_COLOR,
    )

    add_3d_vector(
        figure,
        y,
        "Y = log(partner)",
        PARTNER_COLOR,
    )

    add_3d_vector(
        figure,
        bracket,
        "[X,Y] = X × Y",
        BRACKET_COLOR,
    )

    # # Show that the bracket is orthogonal to X and Y under the R^3 model.
    # figure.add_annotation(
    #     text=(
    #         "For so(3), the Lie bracket is the cross product:<br>"
    #         "[X,Y] = X × Y"
    #     ),
    #     x=0.5,
    #     y=0.02,
    #     xref="paper",
    #     yref="paper",
    #     showarrow=False,
    #     bgcolor="rgba(255,255,255,0.85)",
    # )

    figure.add_annotation(
        text=(
            "Opaque magenta: [X,Y] for selected X and partner Y.<br>"
            "Transparent purple field at Z: "
            f"{COMMUTATOR_FIELD_SCALE:.2f}[Z,X], with selected X fixed."
        ),
        x=0.5,
        y=0.01,
        xref="paper",
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
    )

    maximum = max(
        1.0,
        np.linalg.norm(x),
        np.linalg.norm(y),
        np.linalg.norm(bracket),
    )
    limit = min(maximum * 1.35, 9.0)

    figure.update_layout(
        scene=dict(
            xaxis=dict(title="ω₁", range=[-limit, limit]),
            yaxis=dict(title="ω₂", range=[-limit, limit]),
            zaxis=dict(title="ω₃", range=[-limit, limit]),
            aspectmode="cube",
        )
    )

    return figure


# ============================================================
# Information panel
# ============================================================

def format_vector(vector: np.ndarray) -> str:
    vector = np.asarray(vector, dtype=float)
    return np.array2string(
        vector,
        precision=5,
        suppress_small=True,
        floatmode="fixed",
    )


def format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(
        matrix,
        precision=5,
        suppress_small=True,
        floatmode="fixed",
    )


def make_information_panel(
    data: dict,
    selected_index: int,
    partner_index: int,
    information: dict,
) -> list:
    mode = data["mode"]
    x = information["x"]
    y = information["y"]
    bracket = information["bracket"]
    matrix = information["matrix"]
    inverse_coordinate = information["inverse_coordinate"]

    if mode == "SO3":
        grid_summary = html.P(
            [
                html.Strong("SO(3) grid: "),
                (
                    f"{data.get('actual_count', len(data['coordinates']))} "
                    f"points; requested approximately "
                    f"{data.get('requested_count', len(data['coordinates']))}. "
                    f"Axis-angle coordinate spacing = "
                    f"{data.get('grid_spacing', 0.0):.5f} radians."
                ),
            ]
        )
    else:
        grid_summary = html.P(
            [
                html.Strong("SO(2) samples: "),
                f"{len(data['coordinates'])} points.",
            ]
        )

    bracket_norm = np.linalg.norm(bracket)

    if mode == "SO2":
        explanation = (
            "Every planar rotation commutes with every other planar rotation. "
            "Consequently, so(2) is a one-dimensional Abelian Lie algebra and "
            "its bracket is always zero."
        )
    else:
        explanation = (
            "The principal logarithm converts a rotation into an axis-angle "
            "vector X. Near the identity, these vectors live in a linear "
            "space. Under the standard identification so(3) ≅ ℝ³, the Lie "
            "bracket is X × Y. A nonzero bracket measures the local failure "
            "of the two infinitesimal rotations to commute."
        )

    return [
        html.H3("Selected group element"),
        html.P(
            [
                html.Strong("Selected index: "),
                str(selected_index),
                html.Br(),
                html.Strong("Bracket-partner index: "),
                str(partner_index),
            ]
        ),
        grid_summary,
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Lie-algebra coordinates"),
                        html.Pre(
                            f"X = log(g)\n{format_vector(x)}\n\n"
                            f"Y = log(h)\n{format_vector(y)}\n\n"
                            f"[X,Y]\n{format_vector(bracket)}\n\n"
                            f"||[X,Y]|| = {bracket_norm:.6f}"
                        ),
                    ],
                    className="info-column",
                ),
                html.Div(
                    [
                        html.H4("Group representation"),
                        html.Pre(
                            f"g = exp(X)\n{format_matrix(matrix)}\n\n"
                            f"log(g⁻¹)\n{format_vector(inverse_coordinate)}"
                        ),
                    ],
                    className="info-column",
                ),
            ],
            className="info-grid",
        ),
        html.P(explanation),
    ]


# ============================================================
# Dash application
# ============================================================

app = Dash(__name__)

app.title = "Lie Group and Lie Algebra Explorer"

app.layout = html.Div(
    [
        dcc.Store(id="group-data", data=INITIAL_DATA),

        html.H1("Interactive Lie Group and Lie Algebra Explorer"),

        html.P(
            [
                "Select a group element either from the menu or by clicking "
                "a point in the manifold plot. The selected element is ",
                html.Strong("maize"),
                ", the identity is ",
                html.Strong("green"),
                ", and the inverse is shown with a ",
                html.Strong("blue ×"),
                ".",
            ],
            className="intro-text",
        ),

        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Study case",
                            htmlFor="group-mode",
                            className="control-label",
                        ),
                        dcc.Dropdown(
                            id="group-mode",
                            options=[
                                {
                                    "label": "Simple: SO(2) planar rotations",
                                    "value": "SO2",
                                },
                                {
                                    "label": (
                                        "Complex: axis-angle grid sampled SO(3) "
                                        "rotations"
                                    ),
                                    "value": "SO3",
                                },
                            ],
                            value="SO2",
                            clearable=False,
                        ),
                    ],
                    className="control-item",
                ),

                html.Div(
                    [
                        html.Label(
                            "Selected group element g",
                            htmlFor="selected-element",
                            className="control-label",
                        ),
                        dcc.Dropdown(
                            id="selected-element",
                            options=INITIAL_OPTIONS,
                            value=0,
                            clearable=False,
                            searchable=True,
                        ),
                    ],
                    className="control-item",
                ),

                html.Div(
                    [
                        html.Label(
                            "Bracket partner h",
                            htmlFor="partner-element",
                            className="control-label",
                        ),
                        dcc.Dropdown(
                            id="partner-element",
                            options=INITIAL_OPTIONS,
                            value=1,
                            clearable=False,
                            searchable=True,
                        ),
                    ],
                    className="control-item",
                ),

                html.Div(
                    [
                        html.Label(
                            "SO(3) commutator vector field",
                            htmlFor="show-commutator-field",
                            className="control-label",
                        ),
                        dcc.Checklist(
                            id="show-commutator-field",
                            options=[
                                {
                                    "label": " Show [Z,X] at every grid point",
                                    "value": "show",
                                }
                            ],
                            value=["show"],
                        ),
                    ],
                    className="control-item",
                ),

                html.Div(
                    [
                        html.Label(
                            "Approximate number of SO(3) grid points",
                            htmlFor="so3-point-count",
                            className="control-label",
                        ),
                        dcc.RadioItems(
                            id="so3-point-count",
                            options=[
                                {"label": " ≈20", "value": 20},
                                {"label": " ≈50", "value": 50},
                                {"label": " ≈100", "value": 100},
                                {"label": " ≈200", "value": 200},
                            ],
                            value=100,
                            inline=True,
                            inputStyle={
                                "marginLeft": "10px",
                                "marginRight": "4px",
                            },
                            labelStyle={
                                "display": "inline-block",
                                "marginRight": "8px",
                            },
                        ),
                        html.Small(
                            "Used for SO(3). Complete symmetric grid shells may produce "
                            "slightly more or fewer points than requested.",
                            style={
                                "display": "block",
                                "marginTop": "0.4rem",
                                "color": "#4B5563",
                            },
                        ),
                    ],
                    className="control-item",
                ),

                # html.Div(
                #     [
                #         html.Button(
                #             "Regenerate random SO(3) sample",
                #             id="regenerate-button",
                #             n_clicks=0,
                #             className="regenerate-button",
                #         )
                #     ],
                #     className="button-item",
                # ),
            ],
            className="controls",
        ),

        html.Div(
            [
                dcc.Graph(
                    id="physical-plot",
                    config={"displaylogo": False},
                    className="plot",
                ),
                dcc.Graph(
                    id="manifold-plot",
                    config={"displaylogo": False},
                    className="plot",
                ),
                dcc.Graph(
                    id="algebra-plot",
                    config={"displaylogo": False},
                    className="plot",
                ),
            ],
            className="plot-grid",
        ),

        html.Div(
            id="information-panel",
            className="information-panel",
        ),

        html.Div(
            [
                html.H3("How to interpret the pictures"),
                html.Ul(
                    [
                        html.Li(
                            "Physical space shows what the selected group "
                            "element actually does to an object."
                        ),
                        html.Li(
                            "The manifold plot shows group elements as points "
                            "in a curved or globally constrained space."
                        ),
                        html.Li(
                            "The Lie-algebra plot shows logarithms of group "
                            "elements as vectors in a linear space."
                        ),
                        html.Li(
                            "For SO(3), the axis-angle ball has radius π. "
                            "Antipodal points on the radius-π boundary "
                            "represent the same rotations."
                        ),
                        html.Li(
                            "The Lie algebra is primarily a local model near "
                            "the identity; the logarithm need not give one "
                            "globally unique coordinate for every group "
                            "element."
                        ),
                    ]
                ),
            ],
            className="explanation-panel",
        ),

        # html.Style(
        #     """
        #     body {
        #         margin: 0;
        #         background: #F7F8FA;
        #         color: #1F2937;
        #         font-family: Arial, Helvetica, sans-serif;
        #     }

        #     h1 {
        #         margin-bottom: 0.4rem;
        #     }

        #     .intro-text {
        #         max-width: 1100px;
        #         line-height: 1.55;
        #     }

        #     .controls {
        #         display: grid;
        #         grid-template-columns: repeat(4, minmax(220px, 1fr));
        #         gap: 1rem;
        #         align-items: end;
        #         margin: 1.2rem 0;
        #         padding: 1rem;
        #         background: white;
        #         border: 1px solid #D1D5DB;
        #         border-radius: 10px;
        #     }

        #     .control-label {
        #         display: block;
        #         margin-bottom: 0.4rem;
        #         font-weight: bold;
        #     }

        #     .regenerate-button {
        #         width: 100%;
        #         min-height: 38px;
        #         border: 1px solid #00274C;
        #         border-radius: 6px;
        #         background: #00274C;
        #         color: white;
        #         font-weight: bold;
        #         cursor: pointer;
        #     }

        #     .regenerate-button:hover,
        #     .regenerate-button:focus {
        #         background: #1B4F7A;
        #         outline: 3px solid #FFCB05;
        #     }

        #     .plot-grid {
        #         display: grid;
        #         grid-template-columns: repeat(3, minmax(360px, 1fr));
        #         gap: 1rem;
        #     }

        #     .plot {
        #         min-height: 540px;
        #         background: white;
        #         border: 1px solid #D1D5DB;
        #         border-radius: 10px;
        #     }

        #     .information-panel,
        #     .explanation-panel {
        #         margin-top: 1rem;
        #         padding: 1rem 1.25rem;
        #         background: white;
        #         border: 1px solid #D1D5DB;
        #         border-radius: 10px;
        #         line-height: 1.5;
        #     }

        #     .info-grid {
        #         display: grid;
        #         grid-template-columns: repeat(2, minmax(300px, 1fr));
        #         gap: 1rem;
        #     }

        #     .info-column {
        #         padding: 0.8rem;
        #         background: #F3F4F6;
        #         border-radius: 8px;
        #         overflow-x: auto;
        #     }

        #     pre {
        #         font-size: 0.95rem;
        #         white-space: pre-wrap;
        #     }

        #     body > div > div {
        #         max-width: 1800px;
        #         margin: 0 auto;
        #         padding: 1rem;
        #     }

        #     @media (max-width: 1250px) {
        #         .plot-grid {
        #             grid-template-columns: 1fr;
        #         }

        #         .controls {
        #             grid-template-columns: repeat(2, minmax(220px, 1fr));
        #         }
        #     }

        #     @media (max-width: 700px) {
        #         .controls,
        #         .info-grid {
        #             grid-template-columns: 1fr;
        #         }
        #     }
        #     """
        # ),
    ]
)


# ============================================================
# Callbacks
# ============================================================

# @app.callback(
#     Output("group-data", "data"),
#     Output("selected-element", "options"),
#     Output("selected-element", "value"),
#     Output("partner-element", "options"),
#     Output("partner-element", "value"),
#     Input("group-mode", "value"),
#     Input("regenerate-button", "n_clicks"),
#     Input("manifold-plot", "clickData"),
#     State("group-data", "data"),
#     prevent_initial_call=True,
# )
# def update_group_or_selection(
#     mode: str,
#     regenerate_clicks: int,
#     click_data: dict | None,
#     current_data: dict,
# ):
#     triggering_component = ctx.triggered_id

#     if triggering_component in {"group-mode", "regenerate-button"}:
#         # The button creates a repeatable but different sample on each click.
#         seed = 31415 + 7919 * int(regenerate_clicks or 0)

#         new_data = generate_group_data(mode, seed=seed)
#         options = make_dropdown_options(new_data)

#         # Every new example starts with the identity selected.
#         selected_value = 0
#         partner_value = 1 if len(options) > 1 else 0

#         return (
#             new_data,
#             options,
#             selected_value,
#             options,
#             partner_value,
#         )

#     if triggering_component == "manifold-plot" and click_data:
#         point = click_data.get("points", [{}])[0]
#         custom_data = point.get("customdata")

#         if custom_data is not None:
#             selected_index = int(custom_data)

#             return (
#                 no_update,
#                 no_update,
#                 selected_index,
#                 no_update,
#                 no_update,
#             )

#     return no_update, no_update, no_update, no_update, no_update

@app.callback(
    Output("group-data", "data"),
    Output("selected-element", "options"),
    Output("selected-element", "value"),
    Output("partner-element", "options"),
    Output("partner-element", "value"),
    Input("group-mode", "value"),
    Input("so3-point-count", "value"),
    Input("manifold-plot", "clickData"),
    State("group-data", "data"),
    prevent_initial_call=True,
)
def update_group_or_selection(
    mode: str,
    so3_point_count: int,
    click_data: dict | None,
    current_data: dict,
):
    triggering_component = ctx.triggered_id

    if triggering_component in {
        "group-mode",
        "so3-point-count",
    }:
        new_data = generate_group_data(
            mode=mode,
            number_of_so3_samples=int(so3_point_count or 100),
        )

        options = make_dropdown_options(new_data)

        # Whenever the group or grid density changes, reset the selected
        # element to the identity.
        selected_value = 0
        partner_value = 1 if len(options) > 1 else 0

        return (
            new_data,
            options,
            selected_value,
            options,
            partner_value,
        )

    if triggering_component == "manifold-plot" and click_data:
        point = click_data.get("points", [{}])[0]
        custom_data = point.get("customdata")

        if custom_data is not None:
            selected_index = int(custom_data)

            return (
                no_update,
                no_update,
                selected_index,
                no_update,
                no_update,
            )

    return (
        no_update,
        no_update,
        no_update,
        no_update,
        no_update,
    )


@app.callback(
    Output("physical-plot", "figure"),
    Output("manifold-plot", "figure"),
    Output("algebra-plot", "figure"),
    Output("information-panel", "children"),
    Input("group-data", "data"),
    Input("selected-element", "value"),
    Input("partner-element", "value"),
    Input("show-commutator-field", "value"),
)
def render_explorer(
    data: dict,
    selected_index: int,
    partner_index: int,
    commutator_field_values: list[str],
):
    coordinates = np.asarray(data["coordinates"], dtype=float)
    mode = data["mode"]

    selected_index = int(selected_index or 0)
    partner_index = int(partner_index or 0)

    show_commutator_field = (
        commutator_field_values is not None
        and "show" in commutator_field_values
    )

    information = selected_group_information(
        data,
        selected_index,
        partner_index,
    )

    x = information["x"]
    y = information["y"]
    bracket = information["bracket"]
    inverse_coordinate = information["inverse_coordinate"]

    if mode == "SO2":
        physical_figure = make_so2_physical_figure(x[0])

        manifold_figure = make_so2_manifold_figure(
            coordinates=coordinates,
            selected_index=selected_index,
            partner_index=partner_index,
            inverse_coordinate=inverse_coordinate,
            bracket=bracket,
        )

        algebra_figure = make_so2_algebra_figure(
            x,
            y,
            bracket,
        )
    else:
        physical_figure = make_so3_physical_figure(x)

        manifold_figure = make_so3_manifold_figure(
            coordinates=coordinates,
            selected_index=selected_index,
            partner_index=partner_index,
            inverse_coordinate=inverse_coordinate,
            bracket=bracket,
            show_commutator_field=show_commutator_field,
        )

        algebra_figure = make_so3_algebra_figure(
            x,
            y,
            bracket,
        )

    panel = make_information_panel(
        data,
        selected_index,
        partner_index,
        information,
    )

    return (
        physical_figure,
        manifold_figure,
        algebra_figure,
        panel,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)