    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import  Axes3D
    from sympy.parsing.sympy_parser import parse_expr
    from sympy import plot_implicit
    
    
    def f(x,y):
        return X**2*(X+Y-1)+Y**2*(X+Y-1)
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(-0.1,0.1,0.0001)
    y = np.arange(-0.1,0.1,0.0001)
    X,Y = np.meshgrid(x,y)#创建网格，这个是关键
    Z=f(X,Y)
    plt.contour(X,Y,Z,30) 
    plt.xlabel('x')
    plt.ylabel('y')
    ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
    plt.show()



exc = lambda exper: plot_implicit(parse_expr(exper))
exc('(X**2+Y**2)*(X+Y-1)')  


x=X**2