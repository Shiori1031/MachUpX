"""
将外部软件计算的气动系数 CSV 数据拟合为 MachUpX linear 翼型参数。

用法：
   python fit_airfoil.py

CSV 文件要求：
  - Cl-alpha.csv: 两列，迎角(deg) 和 升力系数 Cl
  - Cd-alpha.csv: 两列，迎角(deg) 和 阻力系数 Cd
  - Cm-alpha.csv: 两列，迎角(deg) 和 力矩系数 Cm
  - 可以带表头也可以不带，脚本自动检测

拟合公式（对应 MachUpX linear 模型）：
  Cl = CLa * (α - aL0)           → 线性拟合 → CLa, aL0
  Cd = CD0 + CD1*Cl + CD2*Cl²    → 二次拟合 → CD0, CD1, CD2
  Cm = CmL0 + Cma * (α - aL0)    → 线性拟合 → Cma, CmL0
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def load_csv(filename):
    """加载 CSV，自动处理表头，返回 (alpha_deg, coef)"""
    data = np.genfromtxt(filename, delimiter=',', dtype=None, encoding='utf-8')
    col1 = data[:, 0]
    col2 = data[:, 1]

    # 检测第一行是否为表头（字符串）
    if isinstance(col1[0], str):
        col1 = col1[1:].astype(float)
        col2 = col2[1:].astype(float)

    return col1, col2


def fit_cl(alpha_rad, cl):
    """线性拟合 Cl = CLa*(α - aL0)"""
    coeffs = np.polyfit(alpha_rad, cl, 1)
    CLa = coeffs[0]
    aL0 = -coeffs[1] / CLa

    cl_fit = CLa * (alpha_rad - aL0)
    r2 = 1 - np.sum((cl - cl_fit)**2) / np.sum((cl - np.mean(cl))**2)

    return CLa, aL0, cl_fit, r2


def fit_cd(cl, cd):
    """拟合 Cd = c + k*(Cl - Cl0)^2，对称轴 Cl0 可调，再转为 CD0/CD1/CD2"""
    # 加重低 Cd 区域权重，让最小阻力点主导拟合
    eps = 1e-8
    weights = 1.0 / (cd**2 + eps)
    weights = weights / weights.sum()

    # Cd = c + k*(Cl^2 - 2*Cl0*Cl + Cl0^2) = (c + k*Cl0^2) + (-2k*Cl0)*Cl + k*Cl^2

    A = np.column_stack([np.ones_like(cl), cl, cl**2])
    W = np.diag(weights)
    coeffs = np.linalg.solve(A.T @ W @ A, A.T @ W @ cd)
    CD0, CD1, CD2 = coeffs[0], coeffs[1], coeffs[2]

    # 对称轴位置
    Cl0 = -CD1 / (2.0 * CD2)

    cd_fit = CD0 + CD1*cl + CD2*cl**2
    r2 = 1 - np.sum((cd - cd_fit)**2) / np.sum((cd - np.mean(cd))**2)

    return CD0, CD1, CD2, cd_fit, r2, Cl0


def fit_cm(alpha_rad, cm, aL0):
    """线性拟合 Cm = CmL0 + Cma*(α - aL0)，输入已是弧度"""
    x = alpha_rad - aL0
    coeffs = np.polyfit(x, cm, 1)
    Cma = coeffs[0]
    CmL0 = coeffs[1]

    cm_fit = CmL0 + Cma * x
    r2 = 1 - np.sum((cm - cm_fit)**2) / np.sum((cm - np.mean(cm))**2)

    return CmL0, Cma, cm_fit, r2


def main():
    script_dir = os.path.dirname(__file__)
    os.chdir(script_dir)

    # 加载数据
    alpha_cl, cl = load_csv("Cl-alpha.csv")
    alpha_cd, cd = load_csv("Cd-alpha.csv")
    alpha_cm, cm = load_csv("Cm-alpha.csv")

    # 只保留 -10° ~ +15° 范围内的数据，并转换为弧度
    mask_cl = (alpha_cl >= -10) & (alpha_cl <= 15)
    alpha_cl, cl = np.radians(alpha_cl[mask_cl]), cl[mask_cl]
    mask_cd = (alpha_cd >= -10) & (alpha_cd <= 15)
    alpha_cd, cd = np.radians(alpha_cd[mask_cd]), cd[mask_cd]
    mask_cm = (alpha_cm >= -10) & (alpha_cm <= 15)
    alpha_cm, cm = np.radians(alpha_cm[mask_cm]), cm[mask_cm]
    # Cm 再进一步限制到 ±0.1 rad
    mask_cm2 = (alpha_cm >= -0.1) & (alpha_cm <= 0.1)
    alpha_cm, cm = alpha_cm[mask_cm2], cm[mask_cm2]

    # ---- 拟合 ----
    CLa, aL0, cl_fit, r2_cl = fit_cl(alpha_cl, cl)
    # Cd 拟合用的 Cl 从 Cl 模型插值得到，保证长度一致
    cl_for_cd = CLa * (alpha_cd - aL0)
    CD0, CD1, CD2, cd_fit, r2_cd, Cl0 = fit_cd(cl_for_cd, cd)
    CmL0, Cma, cm_fit, r2_cm = fit_cm(alpha_cm, cm, aL0)

    # ---- 控制台输出 ----
    print()
    print(f"  Cl = CLa * (alpha - aL0)")
    print(f"  CLa  = {CLa:.4f}  (1/rad)")
    print(f"  aL0  = {np.degrees(aL0):.4f} deg  ({aL0:.6f} rad)")
    print(f"  R^2  = {r2_cl:.6f}")
    print()
    print(f"  Cd = CD0 + CD1*Cl + CD2*Cl^2")
    print(f"  CD0  = {CD0:.6f}")
    print(f"  CD1  = {CD1:.6f}")
    print(f"  CD2  = {CD2:.6f}")
    print(f"  Cl0  = {Cl0:.4f}  (symmetry axis)")
    print(f"  R^2  = {r2_cd:.6f}")
    print()
    print(f"  Cm = CmL0 + Cma * (alpha - aL0)")
    print(f"  CmL0 = {CmL0:.6f}")
    print(f"  Cma  = {Cma:.4f}  (1/rad)")
    print(f"  R^2  = {r2_cm:.6f}")
    print()

    # ---- 画图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Cl vs α
    ax = axes[0]
    ax.plot(alpha_cl, cl, 'ko', markersize=4, label='data')
    ax.plot(alpha_cl, cl_fit, 'r-', label=f'fit Cl={CLa:.2f}(alpha{aL0:+.4f})')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel(r'$\alpha$ (rad)')
    ax.set_ylabel(r'$C_L$')
    ax.set_title(f'Cl - $\\alpha$  (R^2={r2_cl:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cd vs Cl
    ax = axes[1]
    ax.plot(cl_for_cd, cd, 'ko', markersize=4, label='data')
    cl_smooth = np.linspace(cl_for_cd.min(), cl_for_cd.max(), 200)
    cd_smooth = CD0 + CD1*cl_smooth + CD2*cl_smooth**2
    ax.plot(cl_smooth, cd_smooth, 'r-', label=f'fit Cd={CD0:.4f}{CD1:+.4f}Cl{CD2:+.4f}Cl^2')
    ax.axvline(Cl0, color='gray', ls='--', lw=0.5, label=f'Cl0={Cl0:.3f}')
    ax.set_xlabel(r'$C_L$')
    ax.set_ylabel(r'$C_D$')
    ax.set_title(f'Cd - Cl  (R^2={r2_cd:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cm vs α
    ax = axes[2]
    ax.plot(alpha_cm, cm, 'ko', markersize=4, label='data')
    ax.plot(alpha_cm, cm_fit, 'r-', label=f'fit Cm={CmL0:.4f}{Cma:+.4f}(alpha{aL0:+.4f})')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.set_xlabel(r'$\alpha$ (rad)')
    ax.set_ylabel(r'$C_m$')
    ax.set_title(f'Cm - $\\alpha$  (R^2={r2_cm:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
