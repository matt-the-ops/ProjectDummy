"""
Phase 4A-1 verification tests.
Checks:
  1. _cam_tk_size / _right_tk_size sentinel fields exist in __init__
  2. _draw_left_panel reuses cam_tk (paste path) instead of always creating new PhotoImage
  3. _draw_right_panel reuses right_tk (paste path) instead of always creating new PhotoImage
  4. No canvas.delete("all") call exists in _render, _draw_left_panel, _draw_right_panel
  5. bg_image_id still uses itemconfig (not delete/create) in _update_layout_geometry
"""
import ast, sys, unittest

SRC = open("main.py", encoding="utf-8").read()
TREE = ast.parse(SRC)
LINES = SRC.splitlines()

def get_method_src(name, cls=None):
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef):
            for child in ast.walk(node):
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    if cls is None or node.name == cls:
                        return "\n".join(LINES[child.lineno - 1 : child.end_lineno])
    return ""

class TestPhase4A1(unittest.TestCase):

    def test_sentinel_fields_in_init(self):
        """ShieldFSLApp.__init__ must initialise _cam_tk_size and _right_tk_size."""
        init_src = get_method_src("__init__", cls="ShieldFSLApp")
        self.assertIn("_cam_tk_size", init_src,
                      "_cam_tk_size sentinel missing from ShieldFSLApp.__init__")
        self.assertIn("_right_tk_size", init_src,
                      "_right_tk_size sentinel missing from ShieldFSLApp.__init__")

    def test_cam_paste_reuse_path_exists(self):
        """_draw_left_panel must contain the paste() reuse branch."""
        src = get_method_src("_draw_left_panel", cls="ShieldFSLApp")
        self.assertIn("_cam_tk_size", src,
                      "_cam_tk_size size-check missing from _draw_left_panel")
        self.assertIn(".paste(cam_crop)", src,
                      "cam_tk.paste(cam_crop) reuse call missing from _draw_left_panel")

    def test_right_paste_reuse_path_exists(self):
        """_draw_right_panel must contain the paste() reuse branch."""
        src = get_method_src("_draw_right_panel", cls="ShieldFSLApp")
        self.assertIn("_right_tk_size", src,
                      "_right_tk_size size-check missing from _draw_right_panel")
        self.assertIn(".paste(r_img)", src,
                      "right_tk.paste(r_img) reuse call missing from _draw_right_panel")

    def test_no_delete_all_in_render_path(self):
        """canvas.delete('all') must NOT appear in _render, _draw_left_panel, _draw_right_panel."""
        for mname in ("_render", "_draw_left_panel", "_draw_right_panel"):
            src = get_method_src(mname, cls="ShieldFSLApp")
            self.assertNotIn('delete("all")', src,
                             f'canvas.delete("all") found in {mname}')
            self.assertNotIn("delete('all')", src,
                             f"canvas.delete('all') found in {mname}")

    def test_bg_image_uses_itemconfig_not_create(self):
        """_update_layout_geometry must update bg_image_id via itemconfig, not delete+create."""
        src = get_method_src("_update_layout_geometry", cls="ShieldFSLApp")
        self.assertIn("itemconfig(self.bg_image_id", src,
                      "itemconfig(bg_image_id) missing from _update_layout_geometry")
        self.assertNotIn("create_image", src,
                         "create_image found in _update_layout_geometry — should not recreate")

if __name__ == "__main__":
    unittest.main(verbosity=2)
