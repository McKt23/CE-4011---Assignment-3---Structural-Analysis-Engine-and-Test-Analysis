# 🏗️ Advanced Structural Analysis Engine

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Optimized-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Parsing-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)

A professional, object-oriented 2D Finite Element Analysis (FEA) engine built with Python. Developed as a comprehensive solution for complex structural engineering problems, this engine solves 2D Frame and Truss hybrid systems under simultaneous gravity, thermal, and settlement loading conditions.

## ✨ Key Innovations & Features

### 🧠 1. Auto-Stabilizer (Topology Scanner)
Solves the infamous **"Zero Diagonal Paradox"** in Finite Element formulations. Pure truss nodes lack rotational stiffness ($K_{\theta} = 0$), which traditionally causes a singular matrix error if left unrestrained. 
* **How it works:** The engine pre-scans the system topology before matrix assembly. If a node connects exclusively to truss elements (or fully hinged beams), it automatically locks the rotational degree of freedom ($R_z$) in the background. 
* **Result:** Perfect stability for statically determinate trusses without altering true physical boundary conditions or internal forces.

### 🔥 2. Advanced Thermal Load Processing
Superimposes thermal strains with mechanical loads:
* **Uniform Temperature Change ($\Delta T_{avg}$):** Calculates axial expansion/contraction forces.
* **Thermal Gradients ($\Delta T_{grad}$):** Specifically handles differential heating (e.g., bottom heated, top ambient) by calculating fixed-end thermal moments ($M = \frac{E \cdot I \cdot \alpha \cdot \Delta T_{grad}}{h}$), accurately simulating **Thermal Camber** effects.

### 📉 3. Support Settlements
Accurately transforms prescribed spatial displacements (settlements) into equivalent internal force vectors using static condensation and partitioned matrix multiplication.

### 🛡️ 4. Defensive Programming & Smart Filters
* Evaluates input logic before execution (e.g., raises an exception if temperature gradients are applied to truss elements, which cannot bend).
* **Smart Noise Filter:** Automatically detects and ignores $0$-magnitude member loads applied to trusses from empty Excel template cells, preventing false modeling errors.

## 🛠️ Prerequisites & Installation

The engine relies on standard data science libraries for matrix operations and Excel parsing.

```bash
pip install numpy pandas openpyxl
