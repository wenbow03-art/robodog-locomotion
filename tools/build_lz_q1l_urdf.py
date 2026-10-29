"""Build the Isaac Gym asset for LZ-Q1L from the organizer's original URDF.

The original files are left untouched. Output:
    HIMLoco/legged_gym/resources/robots/lz_q1l/{urdf/lz_q1l.urdf, meshes/*.stl}
"""
import os
import shutil
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(
    ROOT, "2026年移动-蓝思杯培训资料", "2026年移动-蓝思杯培训资料", "赛道二开发",
    "02-模型与SDK", "解压", "22nm_dog_创客大赛", "22nm_dog")
DST = os.path.join(ROOT, "HIMLoco", "legged_gym", "resources", "robots", "lz_q1l")

# Joint-side command bounds projected from actuator_params_config.json (lens_kaite: 25 / 30)
# through joint_config.yaml joint_direction (|d| = 1 for abad/hip, 1.5 for knee).
JOINT_LIMITS = {
    "roll": (25.0, 30.0),
    "pitch01": (25.0, 30.0),
    "pitch02": (37.5, 20.0),
}

LEGS = ["f_l", "f_r", "b_l", "b_r"]


def foot_y(leg):
    return -0.01 if leg.endswith("l") else 0.01


# Primitive collision shapes: (type, xyz, rpy, size-attributes)
COLLISIONS = {
    "base_link": [("box", "0.008 0 0", "0 0 0", {"size": "0.50 0.20 0.19"})],
    "head_nod_link": [("box", "0 0 0.048", "0 0 0", {"size": "0.22 0.165 0.15"})],
}
for leg in LEGS:
    side = 1.0 if leg.endswith("l") else -1.0
    COLLISIONS[f"pitch01_leg_{leg}_link"] = [
        ("cylinder", f"-0.0707 {0.0939 * side:.4f} -0.0707", "0 0.7854 0", {"radius": "0.025", "length": "0.14"})]
    COLLISIONS[f"pitch02_leg_{leg}_link"] = [
        ("cylinder", "0.068 0 -0.064", "0 -0.7854 0", {"radius": "0.015", "length": "0.13"})]
    COLLISIONS[f"pitch03_leg_{leg}_link"] = [
        ("sphere", f"-0.006 {foot_y(leg)} -0.008", "0 0 0", {"radius": "0.02"})]


def indent(el, level=0):
    pad = "\n" + "  " * level
    if len(el):
        el.text = pad + "  "
        for child in el:
            indent(child, level + 1)
        child.tail = pad
    if level:
        el.tail = pad


def main():
    tree = ET.parse(os.path.join(SRC, "urdf", "22nm_dog.urdf"))
    robot = tree.getroot()
    robot.set("name", "lz_q1l")

    for tag, name in [("mujoco", None), ("link", "world"), ("joint", "floating_base_joint")]:
        for el in list(robot.findall(tag)):
            if name is None or el.get("name") == name:
                robot.remove(el)

    os.makedirs(os.path.join(DST, "urdf"), exist_ok=True)
    os.makedirs(os.path.join(DST, "meshes"), exist_ok=True)
    for fn in os.listdir(os.path.join(SRC, "meshes")):
        shutil.copyfile(os.path.join(SRC, "meshes", fn), os.path.join(DST, "meshes", fn.lower()))

    for link in robot.findall("link"):
        name = link.get("name")
        for mesh in link.iter("mesh"):
            mesh.set("filename", "../meshes/" + os.path.basename(mesh.get("filename")).lower())
        for col in list(link.findall("collision")):
            link.remove(col)
        for shape, xyz, rpy, attrs in COLLISIONS.get(name, []):
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", {"xyz": xyz, "rpy": rpy})
            geom = ET.SubElement(col, "geometry")
            ET.SubElement(geom, shape, attrs)

    for joint in robot.findall("joint"):
        name = joint.get("name")
        if name.startswith("pitch03_"):
            joint.set("dont_collapse", "true")
        if joint.get("type") == "revolute":
            effort, velocity = JOINT_LIMITS[name.split("_")[0]]
            limit = joint.find("limit")
            limit.set("effort", str(effort))
            limit.set("velocity", str(velocity))

    indent(robot)
    out = os.path.join(DST, "urdf", "lz_q1l.urdf")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
