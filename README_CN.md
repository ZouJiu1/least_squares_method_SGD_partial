[English version README.md](README.md)

<img src="fitline.png">

给定一系列的二维坐标点，这些点在一条直线的旁边，某些落在直线上，某些在直线旁边，某些是离群点

不妨令直线方程是：$y = ax +b$

现在需要求出一条直线方程，让所有坐标点到该直线的最短距离累加和最小，$\text{z}$ = $\sum_{i=1}^{n}{(y_i - y'_i)^2}$

损失函数定义：$\text{z}$ = $\sum_{i=1}^{n}{(y_i - y'_i)^2}$

也就相当是求一个函数sum的极小值，当函数sum取得极小值时，对应的a和b

代入得：$\text{z}$ = $\sum_{i=1}^{n}{[y_i - (a\cdot x_i+b)]^2}$

该函数是三元一次函数，两个自变量$y_i$和$x_i$，一个因变量$\text{z}$，a和b是未知数，需要使用坐标点$(x_i,y_i)$求出a和b

对函数求偏导数：$\frac{\delta\text{z}}{\delta a}=-2\sum_{i=1}^{n}x_i{[y_i - (a\cdot x_i+b)]}$

$\frac{\delta\text{z}}{\delta b}=-2\sum_{i=1}^{n}{[y_i - (a\cdot x_i+b)]}$

## 偏导数=0

按照高等数学里面通常的方案，一般是令偏导数等于0，$\frac{\delta\text{z}}{\delta a}=0$， $\frac{\delta\text{z}}{\delta b}=0$，然后得到两个方程，求出方程的解就可以了

即：$\sum_{i=1}^{n}{x_iy_i}-a\sum_{i=1}^{n}{x^2_i}=b\sum_{i=1}^{n}{x_i}$，和 $\sum_{i=1}^{n}y_i-a\sum_{i=1}^{n}x_i = bn$

令$S_{xy} = \sum_{i=1}^{n}{x_iy_i}$，$S_{xx} = \sum_{i=1}^{n}{x^2_i}$, $S_x=\sum_{i=1}^{n}x_i$, $S_y=\sum_{i=1}^{n}y_i$

得$a\cdot S_{xx}+b\cdot S_x=S_{xy}$ 和 $a\cdot S_x + bn=S_y$

系数行列式$\Delta$：$\Delta=\begin{vmatrix}S_{xx}&S_x\\S_x& n\end{vmatrix}=n\cdot S_{xx}-S^2_x$

根据克拉默法则，求出 $a=\frac{nS_{xy}-S_xS_y}{n\cdot S_{xx}-S^2_x}$, $b=\frac{S_yS_{xx}-S_xS_{xy}}{n\cdot S_{xx}-S^2_x}$

对于比较简单的函数，用偏导数=0是可以求出结果的，对于相当复杂的函数，是求不出解析解的，只能采用其他的方式求出未知数a和b，梯度下降就是其中的一种方式。


## 梯度下降

[https://zoujiu.blog.csdn.net/article/details/159514721?spm=1011.2415.3001.5331](https://zoujiu.blog.csdn.net/article/details/159514721?spm=1011.2415.3001.5331)

此时将a和b看作参数

真实的损失函数：$\text{z}$ = $\sum_{i=1}^{n}{[y_i - (a\cdot x_i+b)]^2}$，考虑到数据太多，内存可能不够，所以一般只用单个数据项的损失函数。

单个数据项的损失函数：$z=loss={[y_i - (a\cdot x_i+b)]^2}$，损失函数越小越好

已知，$\frac{\delta\text{z}}{\delta a}=-2x_i{[y_i - (a\cdot x_i+b)]}$

$\frac{\delta\text{z}}{\delta b}=-2{[y_i - (a\cdot x_i+b)]}$

根据高等数学的内容，梯度是一个向量，$\frac{\delta\text{z}}{\delta a}i$+$\frac{\delta\text{z}}{\delta b}j$，函数沿着梯度方向下降速度最快

此时可以遍历一个数据项，a的变化量是$\frac{\delta\text{z}}{\delta a}$，b的变化量是这个$\frac{\delta\text{z}}{\delta b}$

首先随机给a和b一个初始数值

然后就可以更新参数a和b，$a'=a-lr\cdot \frac{\delta\text{z}}{\delta a}$, $b'=b - lr\cdot \frac{\delta\text{z}}{\delta b}$，lr是learning_rate学习率，一般是比较小的数字像0.0001, 0.001等

再遍历一个数据项，得到a和b的变化量$\frac{\delta\text{z}}{\delta a}=-2x_j{[y_j - (a\cdot x_j+b)]}$，$\frac{\delta\text{z}}{\delta b}=-2{[y_j - (a\cdot x_j+b)]}$

再次更新一点的：$a''=a'-lr\cdot \frac{\delta\text{z}}{\delta a}$, $b''=b' - lr\cdot \frac{\delta\text{z}}{\delta b}$

然后不断遍历其他数据

这样就不需要求出解析解，只需要用数据就可以求出a和b，还可以求出多个数据项的a和b的变化量，然后再更新a和b，变化量只需要累加就可以了

也就是 $\sum_{i=1}^{k}\frac{\delta z}{\delta a_i}$, $\sum_{i=1}^{k}\frac{\delta z}{\delta b_i}$

$a'=a-lr\cdot \sum_{i=1}^{k}\frac{\delta z}{\delta a_i}$, $b'=b - lr\cdot \sum_{i=1}^{k}\frac{\delta z}{\delta b_i}$

## 实例

首先没有噪声，生成了一组坐标点，偏导数计算的结果完全相同；梯度下降计算的结果几乎完全相同

然后生成了一组存在噪声的坐标点，偏导数拟合的效果还可以，梯度下降的结果也类似的

因a和b的数值差距过大，导致a和b学习的速度不相同，a很早就收敛了epoch<1000，但是b一直没有收敛直到epoch>30000以后才开始收敛

所以batchnorm层是用来限制输出的大小，l2正则化weight_decay用来限制参数的大小，防止参数之间学习的速度不相同，导致收敛缓慢

<img src="fig.png">

## 3d坐标点

3d坐标点的方案是类似的，也是偏导数=0，求出三个方程，然后求出对应的系数a, b, c

或者也用梯度下降的方式，写一个小的神经元，迭代求出a, b, c