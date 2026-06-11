import numpy as np
import pandas as pd
import ast         


def get_boundaries(interval_table):
    interval_min = lambda x: ast.literal_eval(x)[0]
    interval_max = lambda x: ast.literal_eval(x)[1]
    f1 = np.vectorize(interval_min)
    f2 = np.vectorize(interval_max)
    lower_boundary = f1(interval_table)
    upper_boundary = f2(interval_table)
    if isinstance(interval_table, pd.DataFrame):
        lower_boundary = pd.DataFrame(lower_boundary, index = interval_table.index, columns = interval_table.columns)
        upper_boundary = pd.DataFrame(upper_boundary, index = interval_table.index, columns = interval_table.columns)
    return lower_boundary, upper_boundary


def get_center_and_radius(interval_table):
    lb, ub = get_boundaries(interval_table = interval_table)
    return (lb + ub)/2, (ub - lb)/2     # center, radius


def check_interval_table(interval_table):
    lb, ub = get_boundaries(interval_table = interval_table)
    if isinstance(lb, pd.DataFrame) and isinstance(ub, pd.DataFrame):
        lb_aux, ub_aux = lb.to_numpy(), ub.to_numpy()
    else:
        lb_aux, ub_aux = lb, ub
    
    if ~np.all(lb_aux <= ub_aux):
        raise ValueError("The dataset is not of the interval type. Lower boundary must be less than or equal to upper boundary.")
    


def from_boundaries_to_interval_table( boundaries, dp = None):
    lb, ub = boundaries

    if isinstance(lb, pd.DataFrame) and isinstance(ub, pd.DataFrame):
        lb_aux, ub_aux = lb.to_numpy(), ub.to_numpy()
    else:
        lb_aux, ub_aux = lb, ub 
    
    if ~np.all(lb_aux <= ub_aux):
        raise ValueError("Lower boundary must be less than or equal to upper boundary.")
    if lb_aux.shape != ub_aux.shape:
        raise ValueError(" The matrices of lower boundaries and upper boundaries must have the same dimensions.")
    
    if dp is not None:
        lb_aux = np.round(lb, dp)
        ub_aux = np.round(ub, dp)

    build_interval = lambda a,b: str([a,b])
    f1 = np.vectorize(build_interval)
    X = f1(lb_aux,ub_aux)

    if isinstance(lb, pd.DataFrame):
        return  pd.DataFrame(X, index=lb.index, columns= lb.columns)
    else:
        return X


def from_centers_and_raddi_to_interval_table(center_and_radius, dp = None):
    center, radius = center_and_radius

    if isinstance(radius, pd.DataFrame):
        radius_aux = radius.to_numpy()
    else:
        radius_aux = radius

    if ~np.all(radius_aux >= 0):
        raise ValueError("The radius cannot be negative.")
    lb = center - radius
    ub = center + radius

    if dp is not None:
        lb = np.round(lb, dp)
        ub = np.round(ub, dp)

    build_interval = lambda a,b: str([a,b])
    f1 = np.vectorize(build_interval)
    X = f1(lb,ub)

    if isinstance(radius, pd.DataFrame):
        return pd.DataFrame(X, index = radius.index, columns = radius.columns)
    else:
        return X 
    

