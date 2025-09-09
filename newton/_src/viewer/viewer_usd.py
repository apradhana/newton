from __future__ import annotations

import os

import numpy as np
import warp as wp

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
except ImportError:
    Gf = Sdf = Usd = UsdGeom = Vt = None

from .viewer import ViewerBase


# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):
    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)


class ViewerUSD(ViewerBase):
    """
    USD viewer backend for Newton physics simulations.

    This backend creates a USD stage and manages mesh prototypes and instanced rendering
    using PointInstancers. It supports time-sampled transforms for efficient playback
    and visualization of simulation data.
    """

    def __init__(self, output_path, fps=60, up_axis="Z", num_frames=None):
        """
        Initialize the USD viewer backend for Newton physics simulations.

        Args:
            output_path (str): Path to the output USD file.
            fps (int, optional): Frames per second for time sampling. Default is 60.
            up_axis (str, optional): USD up axis, either 'Y' or 'Z'. Default is 'Z'.
            num_frames (int, optional): Maximum number of frames to record. If None, recording is unlimited.

        Raises:
            ImportError: If the usd-core package is not installed.
        """
        if Usd is None:
            raise ImportError("usd-core package is required for ViewerUSD. Install with: pip install usd-core")

        super().__init__()

        self.output_path = output_path
        self.fps = fps
        self.up_axis = up_axis
        self.num_frames = num_frames

        # Create USD stage
        self.stage = Usd.Stage.CreateNew(output_path)
        self.stage.SetFramesPerSecond(fps)
        self.stage.SetStartTimeCode(0)

        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        # Track meshes and instancers
        self._meshes = {}  # mesh_name -> prototype_path
        self._instancers = {}  # instancer_name -> UsdGeomPointInstancer
        self._points = {}  # point_name -> UsdGeomPoints

        # Track current frame
        self._frame_index = 0
        self._frame_count = 0

        # Optional mapping for updating existing prim transforms from sim state
        self.path_body_map = None
        self.path_body_relative_transform = None
        self.builder_results = None
        self.parent_translates = None
        self.parent_inv_Rs = None
        self.parent_inv_Rns = None

        self.set_model(None)

    def begin_frame(self, time):
        """
        Begin a new frame at the given simulation time.

        Parameters:
            time (float): The simulation time for the new frame.
        """
        super().begin_frame(time)
        self._frame_index = int(time * self.fps)
        self._frame_count += 1

        # Update stage end time if needed
        if self._frame_index > self.stage.GetEndTimeCode():
            self.stage.SetEndTimeCode(self._frame_index)

    def end_frame(self):
        """
        End the current frame.

        This method is a placeholder for any end-of-frame logic required by the backend.
        """
        pass

    def is_running(self):
        """
        Check if the viewer is still running.

        Returns:
            bool: False if the frame limit is exceeded, True otherwise.
        """
        if self.num_frames is not None:
            return self._frame_count < self.num_frames
        return True

    def close(self):
        """
        Finalize and save the USD stage.

        This should be called when all logging is complete to ensure the USD file is written.
        """
        self.stage.GetRootLayer().Save()
        self.stage = None

        if self.output_path:
            print(f"USD output saved in: {os.path.abspath(self.output_path)}")

    def log_mesh(
        self,
        name,
        points: wp.array,
        indices: wp.array,
        normals: wp.array = None,
        uvs: wp.array = None,
        hidden=False,
        backface_culling=True,
    ):
        """
        Create a USD mesh prototype from vertex and index data.

        Parameters:
            name (str): Mesh name or Sdf.Path string.
            points (wp.array): Vertex positions as a warp array of wp.vec3.
            indices (wp.array): Triangle indices as a warp array of wp.uint32.
            normals (wp.array, optional): Vertex normals as a warp array of wp.vec3.
            uvs (wp.array, optional): UV coordinates as a warp array of wp.vec2.
            hidden (bool, optional): If True, mesh will be hidden. Default is False.
            backface_culling (bool, optional): If True, enable backface culling. Default is True.

        Returns:
            str: The mesh prototype path.
        """

        # Convert warp arrays to numpy
        points_np = points.numpy().astype(np.float32)
        indices_np = indices.numpy().astype(np.uint32)

        if name not in self._meshes:
            self._ensure_scopes_for_path(self.stage, name)

            mesh_prim = UsdGeom.Mesh.Define(self.stage, name)

            # setup topology once (do not set every frame)
            face_vertex_counts = [3] * (len(indices_np) // 3)
            mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(indices_np)

            # Store the prototype path
            self._meshes[name] = mesh_prim

        mesh_prim = self._meshes[name]
        mesh_prim.GetPointsAttr().Set(points_np, self._frame_index)

        # Set normals if provided
        if normals is not None:
            normals_np = normals.numpy().astype(np.float32)
            mesh_prim.GetNormalsAttr().Set(normals_np, self._frame_index)
            mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

        # Set UVs if provided (simplified for now)
        if uvs is not None:
            # TODO: Implement UV support for USD meshes
            pass

        # how to hide the prototype mesh but not the instances in USD?
        # mesh_prim.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    def log_instances(self, name, mesh, xforms, scales, colors, materials):
        """
        Create or update a PointInstancer for mesh instances.

        Parameters:
            name (str): Instancer name or Sdf.Path string.
            mesh (str): Mesh prototype name (must be previously logged).
            xforms (wp.array): Instance transforms as a warp array of wp.transform.
            scales (wp.array): Instance scales as a warp array of wp.vec3.
            colors (wp.array): Instance colors as a warp array of wp.vec3.
            materials (wp.array): Instance materials as a warp array of wp.vec4.

        Raises:
            RuntimeError: If the mesh prototype is not found.
        """
        # Get prototype path
        if mesh not in self._meshes:
            msg = f"Mesh prototype '{mesh}' not found for log_instances(). Call log_mesh() first."
            raise RuntimeError(msg)

        num_instances = len(xforms)

        # Create instancer if it doesn't exist
        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, name)

            instancer = UsdGeom.PointInstancer.Define(self.stage, name)
            instancer.CreateIdsAttr().Set(list(range(num_instances)))
            instancer.CreateProtoIndicesAttr().Set([0] * num_instances)
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            # Set the prototype relationship
            instancer.GetPrototypesRel().AddTarget(mesh)

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        # Convert transforms to USD format
        if xforms is not None:
            xforms_np = xforms.numpy()

            # Extract positions from warp transforms using vectorized operations
            # Warp transform format: [x, y, z, qx, qy, qz, qw]
            positions = xforms_np[:, :3].astype(np.float32)

            # Convert quaternion format: Warp (x, y, z, w) → USD (w, (x,y,z))
            # USD expects quaternions as Gf.Quath(real, imag_vec3)
            quat_w = xforms_np[:, 6].astype(np.float32)
            quat_xyz = xforms_np[:, 3:6].astype(np.float32)

            # Create orientations list with proper USD quaternion format
            orientations = []
            for i in range(num_instances):
                quat = Gf.Quath(
                    float(quat_w[i]), Gf.Vec3h(float(quat_xyz[i, 0]), float(quat_xyz[i, 1]), float(quat_xyz[i, 2]))
                )
                orientations.append(quat)

            # Handle scales with numpy operations
            if scales is None:
                scales = np.ones((num_instances, 3), dtype=np.float32)
            elif isinstance(scales, wp.array):
                scales = scales.numpy().astype(np.float32)

            # Set attributes at current time
            instancer.GetPositionsAttr().Set(positions, self._frame_index)
            instancer.GetOrientationsAttr().Set(orientations, self._frame_index)

            if scales is not None:
                instancer.GetScalesAttr().Set(scales, self._frame_index)

            if colors is not None:
                # Promote colors to proper numpy array format
                colors_np = self._promote_colors_to_array(colors, num_instances)

                # Set color per-instance
                displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                displayColor.Set(colors_np, self._frame_index)

                # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                indices = Vt.IntArray(range(num_instances))
                displayColor.SetIndices(indices, self._frame_index)

    # Abstract methods that need basic implementations
    def log_lines(self, name, starts, ends, colors, width: float = 0.01, hidden=False):
        """Debug helper to add a line list as a set of capsules

        Args:
            starts: The vertices of the lines (wp.array)
            ends: The vertices of the lines (wp.array)
            colors: The colors of the lines (wp.array)
            width: The width of the lines (float)
            hidden: Whether the lines are hidden (bool)
        """

        if name not in self._instancers:
            self._ensure_scopes_for_path(self.stage, name)

            instancer = UsdGeom.PointInstancer.Define(self.stage, name)

            # define nested capsule prim
            instancer_capsule = UsdGeom.Capsule.Define(self.stage, instancer.GetPath().AppendChild("capsule"))
            instancer_capsule.GetRadiusAttr().Set(width)

            instancer.CreatePrototypesRel().SetTargets([instancer_capsule.GetPath()])
            UsdGeom.PrimvarsAPI(instancer).CreatePrimvar(
                "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex, 1
            )

            self._instancers[name] = instancer

        instancer = self._instancers[name]

        if starts is not None and ends is not None:
            num_lines = int(len(starts))
            if num_lines > 0:
                # bring to host
                starts = starts.numpy()
                ends = ends.numpy()

                line_positions = []
                line_rotations = []
                line_scales = []

                for i in range(num_lines):
                    pos0 = starts[i]
                    pos1 = ends[i]

                    (pos, rot, scale) = _compute_segment_xform(
                        Gf.Vec3f(float(pos0[0]), float(pos0[1]), float(pos0[2])),
                        Gf.Vec3f(float(pos1[0]), float(pos1[1]), float(pos1[2])),
                    )

                    line_positions.append(pos)
                    line_rotations.append(rot)
                    line_scales.append(scale)

                instancer.GetPositionsAttr().Set(line_positions, self._frame_index)
                instancer.GetOrientationsAttr().Set(line_rotations, self._frame_index)
                instancer.GetScalesAttr().Set(line_scales, self._frame_index)
                instancer.GetProtoIndicesAttr().Set([0] * num_lines, self._frame_index)
                instancer.CreateIdsAttr().Set(list(range(num_lines)))

                if colors is not None:
                    # Promote colors to proper numpy array format
                    colors_np = self._promote_colors_to_array(colors, num_lines)

                    # Set color per-instance
                    displayColor = UsdGeom.PrimvarsAPI(instancer).GetPrimvar("displayColor")
                    displayColor.Set(colors_np, self._frame_index)

                    # Explicit identity indices [0, 1, 2, ...], otherwise OV won't pick them up
                    indices = Vt.IntArray(range(num_lines))
                    displayColor.SetIndices(indices, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)

    def log_points(self, name, points, radii, colors, hidden=False):
        if np.isscalar(radii):
            radius_interp = "constant"
        else:
            radius_interp = "vertex"

        if colors is None:
            color_interp = "constant"
        elif len(colors) == 3 and all(np.isscalar(x) for x in colors):
            color_interp = "constant"
        else:
            color_interp = "vertex"

        instancer = UsdGeom.Points.Get(self.stage, name)
        if not instancer:
            self._ensure_scopes_for_path(self.stage, name)
            instancer = UsdGeom.Points.Define(self.stage, name)

            UsdGeom.Primvar(instancer.GetWidthsAttr()).SetInterpolation(radius_interp)
            UsdGeom.Primvar(instancer.GetDisplayColorAttr()).SetInterpolation(color_interp)

        instancer.GetPointsAttr().Set(points.numpy(), self._frame_index)

        # convert radii to widths for USD
        if np.isscalar(radii):
            widths = (radii * 2.0,)
        elif isinstance(radii, wp.array):
            widths = radii.numpy() * 2.0
        else:
            widths = np.array(radii) * 2.0

        instancer.GetWidthsAttr().Set(widths, self._frame_index)

        if colors is not None:
            if isinstance(colors, wp.array):
                colors = colors.numpy()
            elif isinstance(colors, list | tuple) and len(colors) == 3:
                colors = (colors,)

            instancer.GetDisplayColorAttr().Set(colors, self._frame_index)

        instancer.GetVisibilityAttr().Set("inherited" if not hidden else "invisible", self._frame_index)
        return instancer.GetPath()

    def log_array(self, name, array):
        """
        Log array data (not implemented for USD backend).

        This method is a placeholder and does not log array data in the USD backend.
        """
        pass

    def log_scalar(self, name, value):
        """
        Log scalar value (not implemented for USD backend).

        This method is a placeholder and does not log scalar values in the USD backend.
        """
        pass

    def _promote_colors_to_array(self, colors, num_items):
        """
        Helper method to promote colors to a numpy array format.

        Parameters:
            colors: Input colors in various formats (wp.array, list/tuple, np.ndarray)
            num_items (int): Number of items that need colors

        Returns:
            np.ndarray: Colors as numpy array with shape (num_items, 3)
        """
        if colors is None:
            return None

        if isinstance(colors, wp.array):
            # Convert warp array to numpy
            return colors.numpy()
        elif isinstance(colors, list | tuple) and len(colors) == 3 and all(np.isscalar(x) for x in colors):
            # Single color (list/tuple of 3 floats) - promote to array with one value per item
            return np.tile(colors, (num_items, 1))
        elif isinstance(colors, np.ndarray):
            # Already numpy array - pass through
            return colors
        else:
            # Fallback for other formats
            return np.array(colors)

    @staticmethod
    def _ensure_scopes_for_path(stage: Usd.Stage, prim_path_str: str):
        """
        Ensure that all parent prims in the hierarchy exist as 'Scope' prims.

        If a prim does not exist at the given path, this method creates all
        non-existent parent prims in its hierarchy as 'Scope' prims. This is
        useful for ensuring a valid hierarchy before defining a prim.

        Parameters:
            stage (Usd.Stage): The USD stage to operate on.
            prim_path_str (str): The Sdf.Path string for the target prim.
        """
        # Convert the string to an Sdf.Path object for robust manipulation
        prim_path = Sdf.Path(prim_path_str)

        # First, check if the target prim already exists.
        if stage.GetPrimAtPath(prim_path):
            return

        # We only want to create the parent hierarchy, not the final prim itself.
        parent_path = prim_path.GetParentPath()

        # GetPrefixes() provides a convenient list of all ancestor paths.
        # For "/A/B/C", it returns ["/", "/A", "/A/B"].
        for path in parent_path.GetPrefixes():
            # The absolute root path ('/') always exists, so we can skip it.
            if path == Sdf.Path.absoluteRootPath:
                continue

            # Check if a prim exists at the current ancestor path.
            if not stage.GetPrimAtPath(path):
                stage.DefinePrim(path, "Scope")

    # ------------------------------
    # Existing-USD-Stage animation API (moved from RendererUsd)
    # ------------------------------

    def attach_source_stage(self, source_stage: str | Usd.Stage, output_stage: str | Usd.Stage):
        """
        Attach a USD stage created from a source stage (flattened copy) as the viewer's stage.

        This mirrors RendererUsd._create_output_stage.
        """
        if isinstance(output_stage, str):
            stage = Usd.Stage.Open(source_stage, Usd.Stage.LoadAll)
            flattened = stage.Flatten()
            temp_stage = Usd.Stage.Open(flattened.identifier)
            exported = temp_stage.ExportToString()

            new_stage = Usd.Stage.CreateNew(output_stage)
            new_stage.GetRootLayer().ImportFromString(exported)
            self.stage = new_stage
        elif isinstance(output_stage, Usd.Stage):
            self.stage = output_stage
        else:
            raise ValueError("output_stage must be a string or a Usd.Stage")

        # carry over fps and basic settings
        self.stage.SetFramesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        return self.stage

    def create_output_stage_from_source(self, source_stage: str | Usd.Stage, output_stage: str | Usd.Stage) -> Usd.Stage:
        """
        Create and return a new stage from a source, without attaching it. Provided for convenience.
        """
        if isinstance(output_stage, str):
            stage = Usd.Stage.Open(source_stage, Usd.Stage.LoadAll)
            flattened = stage.Flatten()
            temp_stage = Usd.Stage.Open(flattened.identifier)
            exported = temp_stage.ExportToString()

            new_stage = Usd.Stage.CreateNew(output_stage)
            new_stage.GetRootLayer().ImportFromString(exported)
            return new_stage
        elif isinstance(output_stage, Usd.Stage):
            return output_stage
        else:
            raise ValueError("output_stage must be a string or a Usd.Stage")

    def configure_body_mapping(
        self,
        path_body_map: dict,
        path_body_relative_transform: dict,
        builder_results: dict,
    ):
        """
        Configure mapping from USD prim paths to body indices and precompute parent inverses.
        """
        if path_body_map is None:
            raise ValueError("path_body_map must be set for configure_body_mapping")
        if path_body_relative_transform is None:
            raise ValueError("path_body_relative_transform must be set for configure_body_mapping")
        if builder_results is None:
            raise ValueError("builder_results must be set for configure_body_mapping")

        self.path_body_map = path_body_map
        self.path_body_relative_transform = path_body_relative_transform
        self.builder_results = builder_results

        # Stage-wide prep to align xform stacks and cache parent inverses
        self._prepare_output_stage_for_mapping()
        self._precompute_parents_xform_inverses()

    def update_stage_from_state(self, state):
        """
        Update USD prim transforms for all mapped prims using the current simulation state.
        Mirrors RendererUsd.render_update_stage/_update_usd_stage.
        """
        if self.path_body_map is None:
            raise ValueError("configure_body_mapping must be called before update_stage_from_state")

        # body_q is either a warp array (preferred) or numpy; ensure numpy here
        body_q = state.body_q.numpy()

        # Use a change block for efficient time-sampled updates
        with Sdf.ChangeBlock():
            for prim_path, body_id in self.path_body_map.items():
                full_xform = body_q[body_id]

                # apply relative xform if any
                rel_xform = self.path_body_relative_transform.get(prim_path)
                if rel_xform:
                    full_xform = wp.mul(full_xform, rel_xform)

                # convert to local space relative to parent (if required)
                full_xform = self._apply_parents_inverse_xform(full_xform, prim_path)

                # set xform ops at current frame index
                self._update_usd_prim_xform(prim_path, full_xform)

    # ---- helpers mirrored from RendererUsd ----

    def _apply_parents_inverse_xform(self, full_xform: wp.transform, prim_path: str) -> wp.transform:
        """
        Map world-space transform (Warp) to the prim's local space using parent's inverse transform pieces.
        """
        current_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
        parent_path = str(current_prim.GetParent().GetPath())

        if parent_path in self.builder_results["path_body_map"]:
            return

        parent_translate = self.parent_translates[parent_path]
        parent_inv_Rot = self.parent_inv_Rs[parent_path]
        parent_inv_Rot_n = self.parent_inv_Rns[parent_path]

        warp_translate = wp.transform_get_translation(full_xform)
        warp_quat = wp.transform_get_rotation(full_xform)

        prim_translate = parent_inv_Rot * (warp_translate - parent_translate)
        prim_quat = parent_inv_Rot_n * warp_quat

        return wp.transform(prim_translate, prim_quat)

    def _update_usd_prim_xform(self, prim_path: str, warp_xform: wp.transform):
        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))

        pos = tuple(map(float, warp_xform[0:3]))
        rot = tuple(map(float, warp_xform[3:7]))

        xform = UsdGeom.Xform(prim)
        xform_ops = xform.GetOrderedXformOps()

        if pos is not None:
            xform_ops[0].Set(Gf.Vec3f(pos[0], pos[1], pos[2]), self._frame_index)
        if rot is not None:
            xform_ops[1].Set(Gf.Quatf(rot[3], rot[0], rot[1], rot[2]), self._frame_index)

    def _compute_parents_inverses(self, prim_path: str, time: Usd.TimeCode):
        prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
        xform = UsdGeom.Xform(prim)

        parent_world = Gf.Matrix4f(xform.ComputeParentToWorldTransform(time))
        Rpw = wp.mat33(parent_world.ExtractRotationMatrix().GetTranspose())
        (_, _, s, _, translate_parent_world, _) = parent_world.Factor()

        transpose_Rpwn = wp.mat33(
            Rpw[0, 0] / s[0],
            Rpw[1, 0] / s[0],
            Rpw[2, 0] / s[0],
            Rpw[0, 1] / s[1],
            Rpw[1, 1] / s[1],
            Rpw[2, 1] / s[1],
            Rpw[0, 2] / s[2],
            Rpw[1, 2] / s[2],
            Rpw[2, 2] / s[2],
        )
        inv_Rpwn = wp.quat_from_matrix(transpose_Rpwn)
        inv_Rpw = wp.inverse(Rpw)

        return translate_parent_world, inv_Rpw, inv_Rpwn

    def _precompute_parents_xform_inverses(self):
        if self.path_body_map is None:
            raise ValueError("path_body_map must be set before calling _precompute_parents_xform_inverses")

        self.parent_translates = {}
        self.parent_inv_Rs = {}
        self.parent_inv_Rns = {}

        time = Usd.TimeCode.Default()
        for prim_path in self.path_body_map.keys():
            current_prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
            parent_path = str(current_prim.GetParent().GetPath())

            if parent_path not in self.parent_translates:
                (
                    self.parent_translates[parent_path],
                    self.parent_inv_Rs[parent_path],
                    self.parent_inv_Rns[parent_path],
                ) = self._compute_parents_inverses(prim_path, time)

    def _prepare_output_stage_for_mapping(self):
        if self.path_body_map is None:
            raise ValueError("path_body_map must be set before calling _prepare_output_stage_for_mapping")

        # Reset time codes to align with viewer timeline
        self.stage.SetStartTimeCode(0.0)
        # keep existing end-time; it will be extended in begin_frame
        self.stage.SetTimeCodesPerSecond(self.fps)

        for prim_path in self.path_body_map.keys():
            prim = self.stage.GetPrimAtPath(Sdf.Path(prim_path))
            self._xform_to_tqs(prim)

    @staticmethod
    def _xform_to_tqs(prim: Usd.Prim, time: Usd.TimeCode | None = None):
        """
        Update the transformation stack of a primitive to translate/orient/scale format.

        The original transformation stack is assumed to be a rigid transformation.
        """
        if time is None:
            time = Usd.TimeCode.Default()

        _tqs_op_order = [UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.TypeScale]
        _tqs_op_precision = [
            UsdGeom.XformOp.PrecisionFloat,
            UsdGeom.XformOp.PrecisionFloat,
            UsdGeom.XformOp.PrecisionFloat,
        ]

        xform = UsdGeom.Xform(prim)
        xform_ops = xform.GetOrderedXformOps()

        # if the order, type, and precision of the transformation is already in our canonical form, then there's no need to change anything.
        if _tqs_op_order == [op.GetOpType() for op in xform_ops] and _tqs_op_precision == [
            op.GetPrecision() for op in xform_ops
        ]:
            return

        # this assumes no skewing
        # NB: the rotation coming from Factor is the result of solving an eigenvalue problem. We found wrong answer with non-identity scaling.
        m_lcl = xform.GetLocalTransformation(time)
        (_, _, scale, _, translation, _) = m_lcl.Factor()

        t = Gf.Vec3f(translation)
        q = Gf.Quatf(m_lcl.ExtractRotationQuat())
        s = Gf.Vec3f(scale)

        # need to reset the transform
        for op in xform_ops:
            attr = op.GetAttr()
            prim.RemoveProperty(attr.GetName())

        xform.ClearXformOpOrder()
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(t)
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(q)
        xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(s)
