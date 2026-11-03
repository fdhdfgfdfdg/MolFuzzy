# MolFuzzy 2.0.7 修正说明

1. 修复可靠性加权下溢：常规计算使用二进制指数缩放，极小乘积使用标准库 Fraction 对输入浮点数做精确乘加，再将权重舍入为 binary64。真正全零的加权输入仍报错。
2. 输出区分原始 R、相对最大值的 R_scaled、二进制尺度 reliability_scale 和原始显示下溢标志。不能用已下溢的 R 重新推算权重。
3. 新增17项测试，覆盖极小正数、同比例缩放、零beta维度、独立Decimal参考、非法输入和完整流程。最终权重本身不可表示时仍可能舍入为零；进入API前已经丢失的信息无法恢复。

测试和数值复现记录见 validation。Windows/macOS未实测；作者声明和公开归档仍待补齐。

已验证：源码222项通过；Linux/Python3.12 + NumPy2.5.4 与 Linux/Python3.9 + NumPy1.24.0 的已安装wheel各222项通过，外部目录CLI JSON通过。
