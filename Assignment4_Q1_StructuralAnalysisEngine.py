import numpy as np
import os
import sys
import tkinter as tk
from tkinter import filedialog

try:
    import pandas as pd
except ImportError:
    pd = None

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# =============================================================================
# 1. CUSTOM EXCEPTIONS
# =============================================================================
class StructuralStabilityError(Exception):
    pass


class ModelingError(Exception):
    pass


class DataFormatError(Exception):
    pass


# =============================================================================
# 2. ENUMS AND DATA CLASSES
# =============================================================================
class ElementType(Enum):
    FRAME = "FRAME"
    TRUSS = "TRUSS"


class Release(Enum):
    NONE = "NONE"
    START = "START"
    END = "END"
    BOTH = "BOTH"


class LoadType(Enum):
    POINT = "POINT"
    DISTRIBUTED = "DISTRIBUTED"


@dataclass
class Node:
    id: int
    x: float
    y: float
    restraints: List[bool] = field(default_factory=lambda: [False, False, False])
    loads: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    settlements: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    eq_nums: List[int] = field(default_factory=lambda: [-1, -1, -1])
    displacements: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class MemberLoad:
    load_type: LoadType
    magnitude: float
    distance: float = 0.0


# =============================================================================
# 3. STRUCTURAL ELEMENT CLASS
# =============================================================================
class Element:
    def __init__(self, id: int, node_i: Node, node_j: Node,
                 E: float, A: float, I: float,
                 elem_type: ElementType = ElementType.FRAME,
                 release: Release = Release.NONE,
                 alpha: float = 1.2e-5, h: float = 0.3,
                 dT_avg: float = 0.0, dT_grad: float = 0.0):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j

        if E <= 0 or A <= 0:
            raise ModelingError(
                f"Element {self.id}: Elastic Modulus (E) or Cross-Sectional Area (A) cannot be zero or negative!")

        self.E = E
        self.A = A
        self.I = I if elem_type == ElementType.FRAME else 1e-12
        self.type = elem_type
        self.release = release if elem_type == ElementType.FRAME else Release.BOTH

        self.alpha = alpha
        self.h = h
        self.dT_avg = dT_avg
        self.dT_grad = dT_grad

        if self.dT_grad != 0.0:
            if self.type == ElementType.TRUSS:
                raise ModelingError(
                    f"Element {self.id}: Physical Contradiction!\n"
                    f"TRUSS elements cannot bend. They only carry axial forces.\n"
                    f"Therefore, a temperature gradient (dT_grad) cannot be applied. You can only assign uniform temperature (dT_avg)."
                )
            if self.type == ElementType.FRAME and self.h <= 0:
                raise ModelingError(
                    f"Element {self.id}: Geometric Deficiency!\n"
                    f"Temperature gradient (dT_grad={self.dT_grad}) is applied, but section height (h) is 0 or negative.\n"
                    f"The 'h' value is mandatory for the thermal moment formula (M = E*I*alpha*dT / h)."
                )

        if self.alpha <= 0 and (self.dT_avg != 0.0 or self.dT_grad != 0.0):
            raise ModelingError(
                f"Element {self.id}: Material Property Error!\n"
                f"A temperature change is applied, but the Thermal Expansion Coefficient (alpha) is invalid.\n"
                f"(Use 1.2e-5 for standard steel)."
            )

        self.member_loads: List[MemberLoad] = []
        self.local_forces = np.zeros(6)

    @property
    def length(self) -> float:
        L = np.hypot(self.node_j.x - self.node_i.x, self.node_j.y - self.node_i.y)
        if L == 0:
            raise ModelingError(f"Element {self.id}: Nodes overlap! Element length cannot be zero.")
        return L

    @property
    def cos_sin(self) -> Tuple[float, float]:
        L = self.length
        return (self.node_j.x - self.node_i.x) / L, (self.node_j.y - self.node_i.y) / L

    def add_load(self, load: MemberLoad):
        if self.type == ElementType.TRUSS:
            raise ModelingError(
                f"Element {self.id} is a TRUSS element. Member/span loads cannot be applied to trusses.")
        self.member_loads.append(load)

    def get_transformation_matrix(self) -> np.ndarray:
        c, s = self.cos_sin
        T = np.zeros((6, 6))
        T[0:3, 0:3] = T[3:6, 3:6] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
        return T

    def _get_raw_stiffness_and_fef(self) -> Tuple[np.ndarray, np.ndarray]:
        L, E, A, I = self.length, self.E, self.A, self.I
        k = np.zeros((6, 6))

        k[0, 0] = k[3, 3] = E * A / L
        k[0, 3] = k[3, 0] = -E * A / L

        k[1, 1] = k[4, 4] = 12 * E * I / L ** 3
        k[1, 4] = k[4, 1] = -12 * E * I / L ** 3
        k[2, 2] = k[5, 5] = 4 * E * I / L
        k[2, 5] = k[5, 2] = 2 * E * I / L
        k[1, 2] = k[2, 1] = k[1, 5] = k[5, 1] = 6 * E * I / L ** 2
        k[4, 2] = k[2, 4] = k[4, 5] = k[5, 4] = -6 * E * I / L ** 2

        fef = np.zeros(6)

        for load in self.member_loads:
            if load.load_type == LoadType.DISTRIBUTED:
                w = load.magnitude
                V = w * L / 2.0
                M = w * L ** 2 / 12.0
                fef += np.array([0, V, M, 0, V, -M])
            elif load.load_type == LoadType.POINT:
                P = load.magnitude
                a = load.distance
                b = L - a
                V1 = P * b ** 2 * (3 * a + b) / L ** 3
                M1 = P * a * b ** 2 / L ** 2
                V2 = P * a ** 2 * (a + 3 * b) / L ** 3
                M2 = -P * a ** 2 * b / L ** 2
                fef += np.array([0, V1, M1, 0, V2, M2])

        P_fef = E * A * self.alpha * self.dT_avg
        M_fef = 0.0
        if self.type == ElementType.FRAME and self.h > 0:
            M_fef = (E * I * self.alpha * self.dT_grad) / self.h

        fef += np.array([P_fef, 0.0, M_fef, -P_fef, 0.0, -M_fef])
        return k, fef

    def _static_condensation(self, k: np.ndarray, fef: np.ndarray, dof_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if k[dof_index, dof_index] == 0: return k, fef

        k_new = np.copy(k)
        fef_new = np.copy(fef)
        k_rr = k[dof_index, dof_index]

        for i in range(6):
            for j in range(6):
                k_new[i, j] = k[i, j] - (k[i, dof_index] * k[dof_index, j]) / k_rr
            fef_new[i] = fef[i] - (fef[dof_index] * k[i, dof_index]) / k_rr

        k_new[dof_index, :] = 0
        k_new[:, dof_index] = 0
        fef_new[dof_index] = 0
        return k_new, fef_new

    def get_condensed_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        k, fef = self._get_raw_stiffness_and_fef()
        if self.release in [Release.START, Release.BOTH]: k, fef = self._static_condensation(k, fef, 2)
        if self.release in [Release.END, Release.BOTH]: k, fef = self._static_condensation(k, fef, 5)
        return k, fef


# =============================================================================
# 4. STRUCTURE ORCHESTRATOR CLASS
# =============================================================================
class Structure:
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.elements: List[Element] = []
        self.num_equations = 0
        self.K_global = None
        self.F_global = None

    def _clean_noise(self, val: float, tol: float = 1e-10) -> float:
        return 0.0 if abs(val) < tol else val

    def add_node(self, node: Node):
        self.nodes[node.id] = node

    def add_element(self, element: Element):
        self.elements.append(element)

    def _apply_auto_stabilization(self):
        """
        TOPOLOGY SCANNER & AUTO-FIX (Advanced Transparent Auto-Stabilization):
        Scans the system topology for 3 critical conditions that mathematically cause singular matrices
        but can be safely auto-locked without altering physical responses (Internal Forces/Reactions).
        """
        stabilized = False

        for node in self.nodes.values():
            # Find all elements connected to this node
            connected_elements = [el for el in self.elements if el.node_i.id == node.id or el.node_j.id == node.id]

            # -------------------------------------------------------------
            # CONDITION 1: Ghost/Orphan Node
            # -------------------------------------------------------------
            if not connected_elements:
                # Lock all DOFs so it doesn't inflate the matrix with zeros
                if not all(node.restraints):
                    node.restraints = [True, True, True]
                    print("\n" + "-" * 80)
                    print(f"🔧 [AUTO-STABILIZER] Auto-Stabilization applied to Node {node.id}")
                    print("   State    : Ghost Node. No elements are connected to this node.")
                    print("   Hazard   : The stiffness matrix would have full zero rows/cols, causing a crash.")
                    print("   Solution : The program safely removed the node from the solver by locking all DOFs.")
                    print("-" * 80)
                    stabilized = True
                continue

            # Evaluate Rotational (Rz) Rigidity
            is_truss_only = all(el.type == ElementType.TRUSS for el in connected_elements)

            is_all_hinged = True
            for el in connected_elements:
                if el.type == ElementType.FRAME:
                    # Check if the frame end connected to THIS node is hinged (released)
                    if (el.node_i.id == node.id and el.release not in [Release.START, Release.BOTH]) or \
                            (el.node_j.id == node.id and el.release not in [Release.END, Release.BOTH]):
                        is_all_hinged = False
                        break

            # -------------------------------------------------------------
            # CONDITION 2: Zero Rotational Stiffness (Trusses or All-Hinged)
            # -------------------------------------------------------------
            if (is_truss_only or is_all_hinged) and not node.restraints[2]:
                node.restraints[2] = True  # Lock Rz
                cause = "Only TRUSS elements" if is_truss_only else "Only HINGED element ends"
                print("\n" + "-" * 80)
                print(f"🔧 [AUTO-STABILIZER] Auto-Stabilization applied to Node {node.id}")
                print(f"   State    : {cause} are connected to this node.")
                print("   Hazard   : Rotational stiffness (Rz) is zero. A singular matrix would occur.")
                print("   Solution : The program automatically locked the Rz restraint (Rz=1).")
                print("   Physics  : Valid. This node receives no moments; locking it does not alter internal forces.")
                print("-" * 80)
                stabilized = True

            # -------------------------------------------------------------
            # CONDITION 3: Collinear Trusses Mechanism
            # -------------------------------------------------------------
            if len(connected_elements) == 2 and is_truss_only:
                el1, el2 = connected_elements[0], connected_elements[1]

                # Check if they are collinear (parallel lines) using cross product
                dx1, dy1 = el1.node_j.x - el1.node_i.x, el1.node_j.y - el1.node_i.y
                dx2, dy2 = el2.node_j.x - el2.node_i.x, el2.node_j.y - el2.node_i.y

                # Cross product near zero implies parallel (collinear)
                cross_product = abs(dx1 * dy2 - dy1 * dx2)

                if cross_product < 1e-5:
                    # They are collinear. Find the direction perpendicular to the truss axis.
                    # If truss is horizontal (dy=0), weak direction is Y. If vertical (dx=0), weak direction is X.
                    weak_dir_index = 1 if abs(dy1) < 1e-5 else (0 if abs(dx1) < 1e-5 else -1)

                    if weak_dir_index != -1 and not node.restraints[weak_dir_index]:
                        # Check if external load in weak direction is ZERO
                        if abs(node.loads[weak_dir_index]) < 1e-5:
                            node.restraints[weak_dir_index] = True
                            dir_name = "Y" if weak_dir_index == 1 else "X"
                            print("\n" + "-" * 80)
                            print(f"🔧 [AUTO-STABILIZER] Auto-Stabilization applied to Node {node.id}")
                            print(
                                f"   State    : Collinear (Inline) Truss Mechanism detected. Load in {dir_name} is zero.")
                            print(
                                f"   Hazard   : Transverse stiffness in {dir_name} direction is zero. Matrix would collapse.")
                            print(f"   Solution : The program safely locked the {dir_name} displacement restraint.")
                            print(
                                f"   Physics  : Valid. Since external load is zero, transverse movement is theoretically zero.")
                            print("-" * 80)
                            stabilized = True

        if stabilized:
            print("\n")

    def _assign_equation_numbers(self):
        eq_count = 0
        for node in self.nodes.values():
            for i in range(3):
                if not node.restraints[i]:
                    node.eq_nums[i] = eq_count
                    eq_count += 1
                else:
                    node.eq_nums[i] = -1
        self.num_equations = eq_count

    def solve(self):
        # Scan topology and execute auto-stabilization algorithms before assembly
        self._apply_auto_stabilization()

        self._assign_equation_numbers()
        N = self.num_equations
        self.K_global = np.zeros((N, N))
        self.F_global = np.zeros(N)

        if N == 0:
            print("\n[INFO] No free Degrees of Freedom (DOF) to solve!")
            print("[INFO] Matrix solution skipped. Only Settlement and Thermal forces will be calculated.")
            D_active = np.array([])
        else:
            for node in self.nodes.values():
                for i in range(3):
                    eq = node.eq_nums[i]
                    if eq != -1: self.F_global[eq] += node.loads[i]

            for el in self.elements:
                k_local, fef_local = el.get_condensed_matrices()
                T = el.get_transformation_matrix()
                k_global_elem = T.T @ k_local @ T
                fef_global_elem = T.T @ fef_local

                d_settlement_global = np.array(el.node_i.settlements + el.node_j.settlements)
                f_settlement_global = k_global_elem @ d_settlement_global

                dof_map = el.node_i.eq_nums + el.node_j.eq_nums

                for i in range(6):
                    eq_i = dof_map[i]
                    if eq_i != -1:
                        self.F_global[eq_i] -= fef_global_elem[i]
                        self.F_global[eq_i] -= f_settlement_global[i]
                        for j in range(6):
                            eq_j = dof_map[j]
                            if eq_j != -1: self.K_global[eq_i, eq_j] += k_global_elem[i, j]

            # Pure Numpy robust solver exception block
            try:
                D_active = np.linalg.solve(self.K_global, self.F_global)
            except np.linalg.LinAlgError:
                raise StructuralStabilityError(
                    "System matrix is SINGULAR (Determinant = 0).\n"
                    "-> Absolute mechanism detected! A node is spinning freely or sliding infinitely.\n"
                    "-> Check your Truss connections and Restraints."
                )

        for node in self.nodes.values():
            for i in range(3):
                eq = node.eq_nums[i]
                if eq != -1 and len(D_active) > 0:
                    node.displacements[i] = D_active[eq]
                else:
                    node.displacements[i] = node.settlements[i]

        for el in self.elements:
            T = el.get_transformation_matrix()
            k_local, fef_local = el.get_condensed_matrices()
            d_global = np.array(el.node_i.displacements + el.node_j.displacements)
            d_local = T @ d_global
            el.local_forces = (k_local @ d_local) + fef_local

    def print_system_matrices(self):
        if self.num_equations == 0: return
        print("\n" + "=" * 80)
        print(" " * 22 + "CALCULATION OUTPUT (MATRICES)")
        print("=" * 80)
        print("\n--- GLOBAL STIFFNESS MATRIX [K] ---")
        np.set_printoptions(suppress=True, precision=2, linewidth=120)
        print(self.K_global)
        print("\n--- GLOBAL FORCE VECTOR {F} ---")
        print(self.F_global)

    def report_results(self):
        print("\n" + "=" * 80)
        print(" " * 24 + "STRUCTURAL ANALYSIS RESULTS")
        print("=" * 80)

        print("\n--- NODAL DISPLACEMENTS ---")
        print(f"{'Node':<6} {'Ux (m)':<15} {'Uy (m)':<15} {'Rz (rad)':<15} {'(Note)'}")
        print("-" * 80)
        for node in self.nodes.values():
            ux = self._clean_noise(node.displacements[0])
            uy = self._clean_noise(node.displacements[1])
            rz = self._clean_noise(node.displacements[2])

            note = "* (Settlement Applied)" if any(node.settlements) and any(node.restraints) else ""
            print(f"{node.id:<6} {ux:<15.6g} {uy:<15.6g} {rz:<15.6g} {note}")

        print("\n--- MEMBER END FORCES (Local Axes) ---")
        print("-" * 80)
        for el in self.elements:
            f = [self._clean_noise(val) for val in el.local_forces]
            print(f"Elem {el.id} ({el.type.name}):")
            print(f"  Start Node {el.node_i.id}: P={f[0]:.3f} kN, V={f[1]:.3f} kN, M={f[2]:.3f} kNm")
            print(f"  End Node   {el.node_j.id}: P={f[3]:.3f} kN, V={f[4]:.3f} kN, M={f[5]:.3f} kNm")

        print("\n--- SUPPORT REACTIONS ---")
        print("-" * 80)
        reactions = {node.id: np.zeros(3) for node in self.nodes.values() if any(node.restraints)}

        for el in self.elements:
            T = el.get_transformation_matrix()
            f_global = T.T @ el.local_forces
            if any(el.node_i.restraints):
                for idx in range(3):
                    if el.node_i.restraints[idx]: reactions[el.node_i.id][idx] += f_global[idx]
            if any(el.node_j.restraints):
                for idx in range(3):
                    if el.node_j.restraints[idx]: reactions[el.node_j.id][idx] += f_global[idx + 3]

        for node in self.nodes.values():
            if node.id in reactions:
                reactions[node.id] -= np.array(node.loads)

        print(f"{'Node':<6} {'Rx (kN)':<15} {'Ry (kN)':<15} {'Mz (kNm)':<15}")
        print("-" * 80)
        for node_id, r in reactions.items():
            rx = self._clean_noise(r[0])
            ry = self._clean_noise(r[1])
            rz = self._clean_noise(r[2])
            print(f"{node_id:<6} {rx:<15.3f} {ry:<15.3f} {rz:<15.3f}")
        print("=" * 80)


# =============================================================================
# 5. SMART EXCEL IMPORTER
# =============================================================================
class ModelImporter:
    @staticmethod
    def parse_bool(val) -> bool:
        if pd.isna(val): return False
        if isinstance(val, (int, float)): return bool(val)
        str_val = str(val).strip().lower()
        return str_val in ['1', '1.0', 'true', 'yes', 'y', 't', 'tutulu', 'sabit', 'fixed', 'fix']

    @staticmethod
    def _clean_columns(df):
        df.columns = [str(col).split('(')[0].strip() for col in df.columns]
        return df

    @staticmethod
    def load_from_excel(filepath: str) -> Structure:
        if pd is None:
            raise ImportError("Pandas and openpyxl are required for Excel operations.")

        xls = pd.ExcelFile(filepath)
        model = Structure()

        df_nodes = ModelImporter._clean_columns(pd.read_excel(xls, 'Nodes'))
        for _, row in df_nodes.iterrows():
            if pd.isna(row.get('ID')): continue

            restraints = [ModelImporter.parse_bool(row.get('Restraint_X', 0)),
                          ModelImporter.parse_bool(row.get('Restraint_Y', 0)),
                          ModelImporter.parse_bool(row.get('Restraint_Rz', 0))]

            settlements = [float(row.get('Settlement_X', 0) if pd.notna(row.get('Settlement_X')) else 0),
                           float(row.get('Settlement_Y', 0) if pd.notna(row.get('Settlement_Y')) else 0),
                           float(row.get('Settlement_Rz', 0) if pd.notna(row.get('Settlement_Rz')) else 0)]

            for i in range(3):
                if (not restraints[i]) and (settlements[i] != 0.0):
                    raise ModelingError(
                        f"Node {int(row['ID'])}: Settlement cannot be defined in a free direction.\n"
                        f"Solution: Set 'Restraint' to 1 for this direction or set Settlement to 0."
                    )

            loads = [float(row.get('Load_X', 0) if pd.notna(row.get('Load_X')) else 0),
                     float(row.get('Load_Y', 0) if pd.notna(row.get('Load_Y')) else 0),
                     float(row.get('Load_M', 0) if pd.notna(row.get('Load_M')) else 0)]

            node = Node(int(row['ID']), float(row.get('X_coord', 0)), float(row.get('Y_coord', 0)), restraints, loads,
                        settlements)
            model.add_node(node)

        df_elems = ModelImporter._clean_columns(pd.read_excel(xls, 'Elements'))
        for _, row in df_elems.iterrows():
            if pd.isna(row.get('ID')): continue
            ni = model.nodes[int(row['Node_I'])]
            nj = model.nodes[int(row['Node_J'])]
            elem_type = str(row.get('Type', 'FRAME')).strip().upper()
            release = str(row.get('Release', 'NONE')).strip().upper()

            alpha = float(row.get('alpha', 1.2e-5)) if pd.notna(row.get('alpha', None)) else 1.2e-5
            h = float(row.get('h', 0.3)) if pd.notna(row.get('h', None)) else 0.3
            dT_avg = float(row.get('dT_avg', 0.0)) if pd.notna(row.get('dT_avg', None)) else 0.0
            dT_grad = float(row.get('dT_grad', 0.0)) if pd.notna(row.get('dT_grad', None)) else 0.0

            elem = Element(int(row['ID']), ni, nj, float(row['E']), float(row['A']), float(row['I']),
                           ElementType[elem_type], Release[release],
                           alpha=alpha, h=h, dT_avg=dT_avg, dT_grad=dT_grad)
            model.add_element(elem)

        if 'MemberLoads' in xls.sheet_names:
            df_loads = ModelImporter._clean_columns(pd.read_excel(xls, 'MemberLoads'))
            for _, row in df_loads.iterrows():
                if pd.isna(row.get('Element_ID')): continue

                # --- SMART NOISE FILTER FOR LOADS ---
                magnitude = float(row.get('Magnitude', 0))
                if abs(magnitude) < 1e-10:
                    continue  # Ignore if load is practically zero (empty cell/placeholder)
                # ------------------------------------

                elem_id = int(row['Element_ID'])
                elem = next((e for e in model.elements if e.id == elem_id), None)
                if elem:
                    load_type_str = str(row.get('Load_Type', 'DISTRIBUTED')).strip().upper()
                    distance = float(row.get('Distance_a', 0) if pd.notna(row.get('Distance_a')) else 0)
                    elem.add_load(MemberLoad(LoadType[load_type_str], magnitude, distance))

        return model


# =============================================================================
# 6. MAIN EXECUTION CONTROLLER
# =============================================================================
if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("=========================================================")
    print("    PROFESSIONAL STRUCTURAL ANALYSIS ENGINE INITIATED    ")
    print("  [Auto-Stabilizer & Smart Filter - V4.9.1]   ")
    print("=========================================================")

    girdi_dosyasi = filedialog.askopenfilename(
        title="Select Model (Excel)",
        filetypes=[("Excel Files", "*.xlsx *.xls")]
    )

    if not girdi_dosyasi:
        sys.exit()

    try:
        my_structure = ModelImporter.load_from_excel(girdi_dosyasi)
        my_structure.solve()

        my_structure.print_system_matrices()
        my_structure.report_results()
        print("\n[SUCCESS] Analysis completed securely and accurately.")

    except StructuralStabilityError as e:
        print("\n" + "!" * 80)
        print(f"🚨 ENGINEERING / STABILITY ERROR 🚨\n{e}")
        print("!" * 80)
    except ModelingError as e:
        print("\n" + "!" * 80)
        print(f"⚠️ MODELING ERROR (LOGIC ERROR) ⚠️\n{e}")
        print("!" * 80)
    except DataFormatError as e:
        print("\n" + "!" * 80)
        print(f"📁 EXCEL DATA FORMAT ERROR 📁\n{e}")
        print("!" * 80)
    except Exception as e:
        print("\n" + "!" * 80)
        print(f"❌ UNEXPECTED SYSTEM ERROR ❌\n{e}")
        print("!" * 80)