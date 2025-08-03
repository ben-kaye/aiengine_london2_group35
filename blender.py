import bpy
import mathutils
from pathlib import Path
from io import StringIO
from contextlib import redirect_stdout
import traceback
import os # Added for path manipulation

def cleanup_scene():
    """
    Removes all objects, materials, images, and other data from the scene
    to ensure a completely clean state for processing the next file.
    """
    # Ensure we are in Object Mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # Delete all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    if bpy.context.selected_objects:
        bpy.ops.object.delete()

    # Purge orphaned data blocks (the most reliable way)
    # This will remove all data that is not used by any object in the scene.
    # Since we just deleted all objects, this will clear everything.
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def _setup_compositor(scene):
    scene.use_nodes = True
    tree = scene.node_tree
    
    for node in tree.nodes:
        tree.nodes.remove(node)
        
    image_node = tree.nodes.new(type='CompositorNodeImage')
    scale_node = tree.nodes.new(type='CompositorNodeScale')
    composite_node = tree.nodes.new(type='CompositorNodeComposite')
    
    scale_node.space = 'RENDER_SIZE'
    scale_node.frame_method = 'FIT'
    
    tree.links.new(image_node.outputs['Image'], scale_node.inputs['Image'])
    tree.links.new(scale_node.outputs['Image'], composite_node.inputs['Image'])
    
    return image_node


def downsample_textures(target_size=(256, 256), output_temp_folder="/tmp/_tmp_textures_"):
    """
    (Corrected) Resizes and saves all images currently in memory.
    """
    out_dir = Path(output_temp_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images to process, excluding non-file types like Render Results.
    images_to_process = [img for img in bpy.data.images if img.type == 'IMAGE']

    if not images_to_process:
        print("No images found to process.")
        return
        
    scene = bpy.context.scene
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y)
    original_filepath = scene.render.filepath
    
    scene.render.resolution_x = target_size[0]
    scene.render.resolution_y = target_size[1]
    scene.render.image_settings.file_format = 'OPEN_EXR'

    image_node = _setup_compositor(scene)
    
    processed_images = []
    for img in images_to_process:
        print(f"Processing '{img.name}'...")
        
        # Use the original filename to avoid "name.exr.exr" issues
        original_name, _ = os.path.splitext(img.name)
        out_path = out_dir / f"{original_name}.exr"
        
        image_node.image = img
        scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        processed_images.append({'original': img, 'new_path': str(out_path)})

    # Override images with their new, downsampled versions
    for data in processed_images:
        img = data['original']
        new_path = data['new_path']
        img.filepath = new_path
        img.source = 'FILE'
        img.reload()
        
    scene.render.resolution_x, scene.render.resolution_y = original_resolution
    scene.render.filepath = original_filepath


def decimate_mesh(obj, ratio=0.1, triangulate=True):
    bpy.context.view_layer.objects.active = obj
    modifier = obj.modifiers.new(name="Decimator", type='DECIMATE')
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = triangulate
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def load_glb(glb_path):
    # This function is the same as before and is correct.
    objects_before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    objects_after = set(bpy.data.objects.keys())
    imported_object_names = objects_after - objects_before

    if not imported_object_names:
        print(f"WARNING: No objects were imported from '{glb_path.name}'. Skipping.")
        return None
        
    imported_objects = [bpy.data.objects[name] for name in imported_object_names]
    imported_meshes = [obj for obj in imported_objects if obj.type == 'MESH']
    other_objects = [obj for obj in imported_objects if obj.type != 'MESH']

    mesh_count = len(imported_meshes)
    has_animation = any(obj.animation_data for obj in imported_objects)
    has_armature = any(mod.type == 'ARMATURE' for obj in imported_objects for mod in obj.modifiers)
    is_invalid = has_animation or has_armature or mesh_count == 0

    if is_invalid:
        print(f"INFO: Rejecting '{glb_path.name}'.")
        # No need to delete objects here, cleanup_scene() will handle it.
        # But we should still delete the invalid source file.
        if glb_path.exists():
             Path(glb_path).unlink()
        return None

    if mesh_count > 1:
        print(f"INFO: Found {mesh_count} meshes in '{glb_path.name}'. Joining them.")
        bpy.ops.object.select_all(action='DESELECT')
        for mesh in imported_meshes:
            mesh.select_set(True)
        bpy.context.view_layer.objects.active = imported_meshes[0]
        bpy.ops.object.join()
        valid_mesh_object = bpy.context.active_object
    else:
        valid_mesh_object = imported_meshes[0]

    if other_objects:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in other_objects:
            obj.select_set(True)
        bpy.ops.object.delete()

    return valid_mesh_object


def center_and_scale_mesh(obj):
    if obj and obj.type == 'MESH':
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0, 0, 0)
        max_dimension = max(obj.dimensions)
        if max_dimension > 0:
            scale_factor = 1.0 / max_dimension
            obj.scale = (scale_factor, scale_factor, scale_factor)
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            print(f"Object '{obj.name}' has been centered and normalized.")
        else:
            print(f"Object '{obj.name}' has no dimensions, cannot normalize.")


def pipe(glb_path):
    """
    (Simplified) Main processing pipeline. No longer needs to delete objects.
    """
    glb_path = Path(glb_path)
    obj = load_glb(glb_path)
    if not obj:
        return

    center_and_scale_mesh(obj)
    decimate_mesh(obj, ratio=0.1, triangulate=True)
    downsample_textures()

    # Select only the processed object for export
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    # Export the processed mesh back to the original path
    bpy.ops.export_scene.gltf(filepath=str(glb_path), use_selection=True)


if __name__ == "__main__":
    log_stream = StringIO()
    log_file_path = Path("~/Downloads/objv_log.txt").expanduser()
    try:
        with redirect_stdout(log_stream):
            # --- MODIFIED MAIN LOOP ---
            for f in Path("~/Downloads/objav-tmp-copy").expanduser().glob("*.glb"):
                print("\n" + "="*50)
                print(f"--- Processing {f.name} ---")
                
                # 1. Clean the scene completely before starting
                cleanup_scene()
                
                # 2. Run the processing pipeline
                pipe(f)

    except Exception as e:
        print("\n" + "=" * 60)
        print("--- SCRIPT FAILED ---")
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc(file=log_stream)
        print("=" * 60)
    finally:
        log_contents = log_stream.getvalue()
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        with log_file_path.open("w") as f:
            f.write(log_contents)
        log_stream.close()

    print(f"✅ Process finished. See log for details: {log_file_path}")