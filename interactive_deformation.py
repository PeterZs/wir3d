### Translation framework instead of quaternions ###
import numpy as np
import os
import torch
import polyscope as ps
import polyscope.imgui as psim
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('objdir', type=str, help='path to obj file')
    parser.add_argument('curvedir', type=str, help='path to curve file')
    parser.add_argument('--meshfeatures', default=None, type=str, help='path to mesh feature file. If not provided, then we use Euclidean distances.')
    parser.add_argument('--curvefeatures', default=None, type=str, help='path to curve feature file. If not provided, then we use Euclidean distances.')
    parser.add_argument('--presampled_t', default=None, type=str, help='Path to torch tensor file of t values. If curvefeatures are provided, then the features must be sampled at these t values.')
    parser.add_argument('--distancetype', choices={'cosine', "l2", "l1"}, default="l2", help='distance type for skinning weights')
    parser.add_argument('--normalization', choices={"linear", "softmax", "mean", "none"}, default="softmax", help='skinning weight normalization')
    parser.add_argument('--texturepath', default=None, type=str, help='path to image texture file, if available.')
    parser.add_argument('--t', type=int, default=10, help="# samples per curve")
    parser.add_argument('--nearestn', type=int, default=0, help="truncate skinning weights to nearest n neighbors. useful for high resolution meshes/many curves.")
    parser.add_argument('--identityweight', type=int, default=0, help="weight on identity deformation / sink parameter to attenuate deformations")

    args = parser.parse_args()

    from pathlib import Path
    # Deformations output directory
    deformations_dir = Path(args.curvedir).parent
    deformations_dir = os.path.join(deformations_dir, "deformations")
    Path(deformations_dir).mkdir(parents=True, exist_ok=True)

    import igl
    
    if args.objdir.endswith(".obj"):
        v, vt, n, f, ftc, _ = igl.readOBJ(args.objdir)
        if len(vt) == 0:
            vt = None
            ftc = None
    else:
        v, f = igl.read_triangle_mesh(args.objdir)
        vt = None
        ftc = None

    vertices = np.array(v, dtype=np.float32)
    faces = np.array(f, dtype=np.int32)
    
    # Normalize in same way as wir3d    
    from igl import bounding_box
    bb_vs, bf = bounding_box(vertices)
    vertices -= np.mean(bb_vs, axis=0)
    vertices /= (np.max(np.linalg.norm(vertices, axis=1)) / 1.0)
    
    curves = torch.load(args.curvedir, weights_only=True, map_location='cpu').float().numpy()

    # Offset curves in the display
    offset = np.array([1, 0, 0.])
    # offset = np.array([0., 0, -5.])
    curves += offset

    # NOTE: Both mesh and curve features must be provided together
    if args.meshfeatures is not None and args.curvefeatures is not None:
        meshfeatures = torch.load(args.meshfeatures, map_location='cpu').numpy().astype(float)
        curvefeatures = torch.load(args.curvefeatures, map_location='cpu').numpy().astype(float)
    else:
        meshfeatures = vertices.astype(float)
        curvefeatures = curves.astype(float) - offset

    if args.presampled_t is not None:
        sample_t = torch.load(args.presampled_t, map_location='cpu').numpy()
        assert torch.all(sample_t >= 0) and torch.all(sample_t <= 1), "Error: presampled t must be between 0 and 1"

        if args.curvefeatures is not None:
            assert len(sample_t) == len(curvefeatures), f"Error: presampled t ({len(sample_t)}) and loaded curvefeatures ({len(curvefeatures)}) must have same length"
    else:
        assert len(curvefeatures) == len(curves), f"Error: curvefeatures ({len(curvefeatures)}) and curves ({len(curves)}) must have same length"
        assert curvefeatures.shape[1] == curves.shape[1], f"Error: curvefeatures ({curvefeatures.shape[1]}) and curves ({curves.shape[1]}) must have same # control points"

        # Sample bezier points using t
        sample_t = np.linspace(0, 1, args.t)

    # NOTE: bezier interpolation follows binomial distribution so the weights will always sum to 1
    w0 = (1 - sample_t)**3
    w1 = 3 * (1 - sample_t)**2 * sample_t
    w2 = 3 * (1 - sample_t) * sample_t**2
    w3 = sample_t**3

    # Get interpolated points
    presampled_points = w0.reshape(1, -1, 1) * curves[:, None, 0] + \
                w1.reshape(1, -1, 1) * curves[:, None, 1] + \
                w2.reshape(1, -1, 1) * curves[:, None, 2] + \
                    w3.reshape(1, -1, 1) * curves[:, None, 3]

    # Get interpolated features (presampled t has pre-interpolated features)
    if args.presampled_t is None or args.curvefeatures is None:
        curvefeatures = w0.reshape(1, -1, 1) * curvefeatures[:, None, 0] + \
                    w1.reshape(1, -1, 1) * curvefeatures[:, None, 1] + \
                    w2.reshape(1, -1, 1) * curvefeatures[:, None, 2] + \
                        w3.reshape(1, -1, 1) * curvefeatures[:, None, 3]

    distancetype = args.distancetype

    # Initialize skinning weights and control point deformations
    def soft_nearest_neighbors_curves(vertices, query_points, t = 10, eps=1e-8, normalization='mean',
                           distancetype='l2', n=0, identityweight=0, normalize_each_curve=False):
        # Returns V x Q matrix of soft weights
        # NOTE: Positive distances only
        if distancetype == "l2":
            dists = 1 / (torch.cdist(vertices, query_points) + eps)
        elif distancetype == "l1":
            dists = 1 / (torch.cdist(vertices, query_points, p=1) + eps)
        elif distancetype == "cosine":
            dists = torch.nn.functional.cosine_similarity(vertices.unsqueeze(1), query_points.unsqueeze(0), dim=2) # [-1, 1]
            dists = (dists + 1) / 2
        else:
            raise ValueError(f"soft_nearest_neighbors: Invalid distance type {distancetype}")

        # Reshape to V x Curves x t
        if normalize_each_curve:
            dists = dists.view(dists.shape[0], -1, t)
            dim = 2
        else:
            dim = 1

        if identityweight > 0:
            dists = torch.cat([dists, torch.ones_like(dists[:, :, :1]) * identityweight], dim=dim)

        assert torch.all(torch.isfinite(dists)), "soft_nearest_neighbors: Non-finite distances"
        # assert torch.all(dists >= 0), "soft_nearest_neighbors: Negative distances"

        # Truncation setting (closest N neighbors)
        if n > 0:
            topdists, indices = torch.topk(dists, n, dim=dim)

            if normalization == "softmax":
                topdists = torch.nn.functional.softmax(topdists, dim=dim)
            elif normalization == "linear":
                topdists = topdists / torch.sum(topdists, dim=dim, keepdim=True)
            elif normalization == "mean":
                topdists = topdists / n

            # Replace the topk values with the softmaxed values and zero out the rest
            dists.fill_(0)
            dists.scatter_(2, indices, topdists)

            return dists
        else:
            if normalization == 'softmax':
                return torch.nn.functional.softmax(dists, dim=dim)
            elif normalization == "linear":
                return dists / torch.sum(dists, dim=dim, keepdim=True)
            elif normalization == "mean":
                return dists / dists.shape[dim]
            else:
                return dists
            
    sampled_nn_weights = soft_nearest_neighbors_curves(torch.from_numpy(meshfeatures).float(),
                                                torch.from_numpy(curvefeatures).reshape(-1, meshfeatures.shape[1]).float(),
                                                args.t,
                                                normalization=args.normalization,
                                                n=args.nearestn, identityweight=args.identityweight,
                                                distancetype=distancetype,
                                                )  # V x (len(curves) * linspace + 1)
    sampled_nn_weights = sampled_nn_weights.reshape(len(vertices), -1)

    ######## Deformations ########
    control_def = torch.tensor([0.,0.,0.])[None, None].repeat(len(curves), 4, 1)

    # Interpolate the quantities
    w = torch.from_numpy(np.stack([w0, w1, w2, w3], axis=1)) # linspace x 4
    def_samples = torch.sum(control_def[:, None, :, :] * w[None, :, :, None], dim=2) # len(curves) x linspace x 3

    # Assign to vertices using the vertex nn weights
    if args.identityweight > 0:
        def_identity_samples = torch.cat([def_samples.reshape(-1, 3), torch.zeros((1, 3))], dim=0) # (len(curves) * linspace + 1) x 3
    else:
        def_identity_samples = def_samples.reshape(-1, 3) # len(curves) * linspace x 3
        
    def_vertex_samples = torch.sum(sampled_nn_weights.unsqueeze(-1) * def_identity_samples[None], dim=1) # V x 3

    prev_selected_curve_index = None
    prev_selected_control_index = None
    curve_index = None
    control_index = None

    # Def and transcurves
    defcurves = np.copy(curves)
    transcurves = np.copy(curves)

    # Translations for determining deformation
    x_def = 0
    y_def = 0
    z_def = 0
    def_cache = {}
    # Translations for determining curve influence
    x_translation = 0
    y_translation = 0
    z_translation = 0
    translations_cache = {}
    x_defchange, y_defchange, z_defchange = False, False, False
    def callback():
        # If we want to use local variables & assign to them in the UI code below,
        # we need to mark them as nonlocal. This is because of how Python scoping
        # rules work, not anything particular about Polyscope or ImGui.
        # Of course, you can also use any other kind of python variable as a controllable
        # value in the UI, such as a value from a dictionary, or a class member. Just be
        # sure to assign the result of the ImGui call to the value, as in the examples below.
        #
        # If these variables are defined at the top level of a Python script file (i.e., not
        # inside any method), you will need to use the `global` keyword instead of `nonlocal`.

        global args, ps_mesh, ps_curve, ps_control_active, vertices, faces
        global control_def, x_def, y_def, z_def
        global transcurves, defcurves, x_translation, y_translation, z_translation
        global prev_selected_curve_index, prev_selected_control_index, translations_cache, def_cache
        global curve_index, control_index
        global x_defchange, y_defchange, z_defchange

        global distancetype, w, w0, w1, w2, w3, presampled_points
        global meshfeatures, curvefeatures, sampled_nn_weights

        vrange = np.arange(len(vertices))
        frange = np.arange(len(vertices), len(vertices) + len(faces))

        # == Settings
        # Use settings like this to change the UI appearance.
        # Note that it is a push/pop pair, with the matching pop() below.
        psim.PushItemWidth(150)

        # == Title window
        psim.TextUnformatted("WIR3D Interactive Deformations")
        psim.Separator()

        # == Reset button ==
        if(psim.Button("Reset")):
            ## Reset the selections ##
            # ps.set_selection("mesh", 0)

            prev_selected_curve_index = None
            prev_selected_control_index = None
            curve_index = None
            control_index = None
            transcurves = np.copy(curves)
            defcurves = np.copy(curves)
            x_def = 0
            y_def = 0
            z_def = 0
            def_cache = {}

            x_translation = 0
            y_translation = 0
            z_translation = 0
            translations_cache = {}

            ## Reset the mesh and deformation quantities ##
            control_def = torch.tensor([0.,0.,0.])[None, None].repeat(len(curves), 4, 1)

            for i in range(len(presampled_points)):
                ps_curve = ps.get_curve_network(f"curve_{i}")
                ps_curve.update_node_positions(presampled_points[i])
                ps_curve.set_color((0, 0, 0))

            ps_control_active.set_enabled(False)

            ps_mesh.update_vertex_positions(vertices)
            ps_mesh.remove_quantity("influence")

        if(psim.Button("Export")):
            # Save the deformed curves
            np.save(os.path.join(deformations_dir, "deformed_curves.npy"), defcurves)

            # Save the deformation cache
            import dill as pickle
            with open(os.path.join(deformations_dir, "defcache.pkl"), 'wb') as f:
                pickle.dump(def_cache, f)

            # Save the deformation quantities (conditioned on t)
            control_samples = torch.sum(control_def[:, None, :, :] * w[None, :, :, None], dim=2) # len(curves) x linspace x 3
            # Convert to identity samples
            if args.identityweight > 0:
                control_identity_samples = torch.cat([control_samples.reshape(-1, 3), torch.zeros((1, 3))], dim=0) # (len(curves) * linspace + 1) x 3
            else:
                control_identity_samples = control_samples.reshape(-1, 3)  # len(curves) * linspace x 3
            np.save(os.path.join(deformations_dir, "deformed_quantities.npy"), control_identity_samples.detach().cpu().numpy())

            # Save the skinning weights
            np.save(os.path.join(deformations_dir, "skinning_weights.npy"), sampled_nn_weights.detach().cpu().numpy())

            control_vertex_samples = torch.sum(sampled_nn_weights.unsqueeze(-1) * control_identity_samples[None], dim=1) # V x 3
            pred_vertices = torch.from_numpy(vertices).float() + control_vertex_samples
            pred_vertices = pred_vertices.detach().cpu().numpy()

            # Save the deformed mesh
            igl.write_triangle_mesh(os.path.join(deformations_dir, "deformed_mesh.obj"), pred_vertices, faces)

        psim.Separator()

        # == On click event, highlight the selected curve/control points ==
        ### Keep track of the clicked curves/control points
        # NOTE: Structure gives you the string name of the structure
        selection = ps.get_selection()
        structure = selection.structure_name
        index = selection.local_index
        structure_data = selection.structure_data

        if "control" in structure:
            # NOTE: Curve index should already be defined if control point is active
            control_index = index

            if prev_selected_control_index is not None:
                if control_index != prev_selected_control_index:
                    # NOTE: If either index changes, then we need to update the control point color
                    newcolors = np.zeros((ps_control_active.n_points(), 3))
                    newcolors[:] = (1, 1, 0)
                    newcolors[control_index] = (1, 0, 0)
                    ps_control_active.add_color_quantity("color", newcolors, enabled=True)

                    prev_selected_control_index = control_index

                    # Set def values based on the cache
                if (curve_index, control_index) not in def_cache:
                    def_cache[(curve_index, control_index)] = (0, 0, 0)
                    x_translation, y_translation, z_translation = 0, 0, 0
                else:
                    x_translation, y_translation, z_translation = def_cache[(curve_index, control_index)]

            else:
                newcolors = np.zeros((ps_control_active.n_points(), 3))
                newcolors[:] = (1, 1, 0)
                newcolors[control_index] = (1, 0, 0)
                ps_control_active.add_color_quantity("color", newcolors, enabled=True)

                prev_selected_control_index = control_index
            
                # Set def values based on the cache
                if (curve_index, control_index) not in def_cache:
                    def_cache[(curve_index, control_index)] = (0, 0, 0)
                    x_translation, y_translation, z_translation = 0, 0, 0
                else:
                    x_translation, y_translation, z_translation = def_cache[(curve_index, control_index)]

                # Set mesh colors based on the control point influence
                # curve_influence = torch.sum(sampled_nn_weights[:, curve_index * args.t: (curve_index + 1) * args.t], dim=1)
                # ps_mesh.add_scalar_quantity("influence", curve_influence.detach().cpu().numpy(), defined_on='vertices', cmap='reds', enabled=True)

        elif "curve" in structure:
            curve_index = int(structure.split("_")[-1])
            control_index = None
            ps_control_active.set_enabled(True)

            if prev_selected_curve_index is not None:
                if curve_index != prev_selected_curve_index:
                    # Reset the color of the previously selected curve
                    prev_curve = ps.get_curve_network(f"curve_{prev_selected_curve_index}")
                    prev_curve.set_color((0, 0, 0))

                    # Set the new active control point positions
                    # Reset the color of the previously selected control points
                    ps_control_active.update_point_positions(curves[curve_index])
                    newcolors = np.zeros((ps_control_active.n_points(), 3))
                    newcolors[:] = (1, 1, 0)
                    ps_control_active.add_color_quantity("color", newcolors, enabled=True)

                # Set new selection to yellow
                ps_curve = ps.get_curve_network(f"curve_{curve_index}")
                ps_curve.set_color((1, 1, 0))

                prev_selected_curve_index = curve_index
                prev_selected_control_index = None

                if (curve_index, control_index) not in def_cache:
                    def_cache[(curve_index, control_index)] = (0, 0, 0)
                    x_translation, y_translation, z_translation = 0, 0, 0
                else:
                    x_translation, y_translation, z_translation = def_cache[(curve_index, control_index)]

                # Set mesh colors based on the control point influence
                # curve_influence = torch.sum(sampled_nn_weights[:, curve_index * args.t: (curve_index + 1) * args.t], dim=1)
                # ps_mesh.add_scalar_quantity("influence", curve_influence.detach().cpu().numpy(), defined_on='vertices', cmap='reds', enabled=True)

            else:
                # Set new selection to yellow
                ps_curve = ps.get_curve_network(f"curve_{curve_index}")
                ps_curve.set_color((1, 1, 0))

                prev_selected_curve_index = curve_index
                prev_selected_control_index = None

                 # Set the new active control point positions
                # Reset the color of the previously selected control points
                ps_control_active.update_point_positions(curves[curve_index])
                newcolors = np.zeros((ps_control_active.n_points(), 3))
                newcolors[:] = (1, 1, 0)
                ps_control_active.add_color_quantity("color", newcolors, enabled=True)

                if (curve_index, control_index) not in def_cache:
                    def_cache[(curve_index, control_index)] = (0, 0, 0)
                    x_translation, y_translation, z_translation = 0, 0, 0
                else:
                    x_translation, y_translation, z_translation = def_cache[(curve_index, control_index)]

        x_change, x_translation = psim.SliderFloat("X", x_translation, v_min=-1, v_max=1)
        y_change, y_translation = psim.SliderFloat("Y", y_translation, v_min=-1, v_max=1)
        z_change, z_translation = psim.SliderFloat("Z", z_translation, v_min=-1, v_max=1)

        psim.Separator()

        if (x_change or y_change or z_change) and curve_index is not None:
            # Get the new deformations
            if (curve_index, control_index) in def_cache:
                prev_def = torch.tensor(def_cache[(curve_index, control_index)])
            else:
                prev_def = torch.tensor([0, 0, 0])

            # NOTE: Only valid for deformation change
            new_translation = torch.tensor([x_translation, y_translation, z_translation])
            if control_index is not None:
                control_def[curve_index, control_index] += (new_translation - prev_def)
            else:
                control_def[curve_index] += (new_translation - prev_def)

            control_samples = torch.sum(control_def[:, None, :, :] * w[None, :, :, None], dim=2) # len(curves) x linspace x 3
            # control_sample = torch.sum(control_def[curve_index, None, :, :] * w[:, :, None], dim=1) # linspace x 3

            # Convert to identity samples
            if args.identityweight > 0:
                control_identity_samples = torch.cat([control_samples.reshape(-1, 3), torch.zeros((1, 3))], dim=0) # (len(curves) * linspace + 1) x 3
                # control_identity_sample = torch.cat([control_sample, torch.zeros((1, 3))], dim=0) # linspace + 1 x 3
            else:
                control_identity_samples = control_samples.reshape(-1, 3)  # len(curves) * linspace x 3
                # control_identity_sample = control_sample  # linspace x 3

            # NOTE: Only compute the samples over the selected curve! (re-linearize by curve)
            # sampled_nn_weight = sampled_nn_weights[:, curve_index * args.t: (curve_index + 1) * args.t] # V x linspace
            # if args.identityweight > 0:
            #     sampled_nn_weight = torch.cat([sampled_nn_weight, sampled_nn_weights[:, [-1]]], dim=1)
            # sampled_nn_weight = torch.softmax(sampled_nn_weight, dim=1)
            # sampled_nn_weight = sampled_nn_weight / torch.linalg.norm(sampled_nn_weight, dim=1, keepdim=True)
            # control_vertex_samples = torch.sum(sampled_nn_weight.unsqueeze(-1) * control_identity_sample[None], dim=1) # V x 3
            control_vertex_samples = torch.sum(sampled_nn_weights.unsqueeze(-1) * control_identity_samples[None], dim=1) # V x 3
            pred_vertices = torch.from_numpy(vertices).float() + control_vertex_samples

            ps_mesh.update_vertex_positions(pred_vertices.detach().cpu().numpy())

            ## NOTE: Update cache and curves AFTER computing the deformation
            ## If control point is selected, then update the control point
            newcontrols = defcurves[curve_index]

            # Gives the previous def values
            if control_index is not None:
                # Update the control point position
                newcontrols[control_index] += (np.array([x_translation, y_translation, z_translation]) - prev_def.numpy())
            else:
                # Update the control point position
                newcontrols += (np.array([x_translation, y_translation, z_translation]) - prev_def.numpy())
            ps_control_active.update_point_positions(defcurves[curve_index])

            # Update the curve
            ps_curve = ps.get_curve_network(f"curve_{curve_index}")

            curve_presampled_points = w0.reshape(-1, 1) * newcontrols[None, 0] + \
                            w1.reshape(-1, 1) * newcontrols[None, 1] + \
                            w2.reshape(-1, 1) * newcontrols[None, 2] + \
                                w3.reshape(-1, 1) * newcontrols[None, 3] # T x 3
            ps_curve.update_node_positions(curve_presampled_points)

            def_cache[(curve_index, control_index)] = (x_translation, y_translation, z_translation)

            # Debugging: print the rotation and scale assigned to vertex 613
            # print(f"Vertex rotation/scale", vertex_rotations[613], "\n", q_scale_samples[613])
            # if control_index is not None:
            #     print("Original control", oldpoint)
            #     print("Control rot/scale", newrot, "\n", newscale)
            #     print("Deformed control", (newrot * newscale) @ oldpoint)
            #     print("New control", newpoint)
            #     print(f"Original vertex", vertices[613])
            #     print(f"Raw deformation applied", (newrot * newscale) @ vertices[613])
            # print("===========================")


        psim.PopItemWidth()

    ps.init()
    ps.remove_all_structures()
    ps.set_ground_plane_mode("none")

    ps_mesh = ps.register_surface_mesh("mesh", vertices, faces)

    if args.texturepath is not None and vt is not None:
        from PIL import Image
        texture = Image.open(args.texturepath)
        texture = np.array(texture)
        texture = texture / 255.0

        if ftc is not None:
            fuv = vt[ftc].reshape(-1, 2)
        else:
            fuv = vt[f].reshape(-1, 2)
            
        ps_mesh.add_parameterization_quantity("uv", fuv, defined_on='corners', enabled=True)
        ps_mesh.add_color_quantity("texture", texture, defined_on='texture', enabled=True, param_name='uv')

    for i in range(len(presampled_points)):
        ps_curve = ps.register_curve_network(f"curve_{i}", presampled_points[i], edges='line', enabled=True, color=(0, 0, 0))
        ps_curve.set_radius(0.01, relative=False)
        ps_curve.set_color((0, 0, 0))

        # Control points
        # ps_control = ps.register_point_cloud(f"control_{i}", curves[i], enabled=True, color=(0, 0, 0))
        # ps_control.set_radius(0.015, relative=False)
        # ps_control.set_color((0, 0, 0))

    # Active control point
    ps_control_active = ps.register_point_cloud(f"control_active", np.zeros((4, 3)), enabled=False, color=(0, 0, 0))
    ps_control_active.set_radius(0.015, relative=False)

    ps.set_invoke_user_callback_for_nested_show(True)
    ps.set_user_callback(callback)
    ps.show()