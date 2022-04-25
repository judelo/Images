
######
## Functions common to all methods
######

def TVaffected(x):
    return np.abs(np.roll(x,1,axis=0)- x) + np.abs(np.roll(x,1,axis=1) - x) + np.abs(np.roll(x,-1,axis=0)- x) + np.abs(np.roll(x,-1,axis=1) - x)


def transition_all_TV(x,alpha, delta_pointwise, **opts): 
    nr,nc = x.shape
    # to parallelize the process we will select a grid of independent pixels with distance 2 between them.
    # select init pixel to draw a grid of independent pixels
    i=np.random.randint(2) 
    j=np.random.randint(2)
    grid = np.zeros((nr,nc))
    grid[i:nr:2,j:nc:2] = 1

    # xtemp =  x + t for pixels of the grid. xtmep = x otherwise.   
    t = np.random.uniform(-alpha, alpha, size=(nr,nc))
    xtemp = np.copy(x) 
    xtemp[grid==1] = x[grid==1] + t[grid==1]

    # decide for each point if we apply the transition or not
    z = np.random.uniform(0, 1, size=(nr,nc))
    diff = np.exp(-delta_pointwise(x, xtemp, **opts) ) - z
    xtemp[diff<0] = x[diff<0]
    return xtemp

######
## Sampling TV: SOLUTION
######

def deltaTV_pointwise(x,xtemp, **opts):
    epsilon = opts.get('epsilon', 1e-4)
    beta = opts.get('beta', 1)
    return  beta*( TVaffected(xtemp) - TVaffected(x)) + epsilon*(xtemp**2-x**2) 


def metropolis_TV1(x,alpha,N, **opts):
    # Metropolis algorithm
    for t in range(N):
        x=transition_all_TV(x,alpha,deltaTV_pointwise, **opts)
    return x

######
## Sampling TVL2: SOLUTION
######

def deltaTVL2_pointwise(x,xtemp, **opts):
    ub = opts.get('ub', x)
    sigma = opts.get('sigma', 1)
    lambd = opts.get('lambd', 1)
    return  lambd*( TVaffected(xtemp) - TVaffected(x)) + 0.5*((xtemp-ub)**2- (x-ub)**2)/(sigma**2) 


def metropolis_TVL2(x,alpha,N, **opts):
    t_burnin = int(N/10)
    nr,nc = x.shape
    xmean = np.zeros((nr,nc))
    x2 = np.zeros((nr,nc))
    # Metropolis algorithm
    for t in range(N):
        x= transition_all_TV(x,alpha,deltaTVL2_pointwise, **opts)
        # update the mean
        if t >= t_burnin:
            tb = t - t_burnin
            xmean = tb/(tb+1)*xmean + 1/(tb+1)*x
            x2    = tb/(tb+1)*x2 + 1/(tb+1)*x**2
    stdx = np.sqrt(x2 - xmean**2)
    return x,xmean,stdx

######
## Sampling TV1L2 with diagonal A 
######

def deltaTVL2A_pointwise(x,xtemp, **opts):
    mask = opts.get('mask', np.ones_like(x))
    ub = opts.get('ub', x)
    sigma = opts.get('sigma', 1)
    lambd = opts.get('lambd', 1)
    return  lambd*( TVaffected(xtemp) - TVaffected(x)) + 0.5*((mask*xtemp-ub)**2- (mask*x-ub)**2)/(sigma**2) 
    
def metropolis_TVL2A(x,alpha,N,**opts):
    t_burnin = int(N/10)
    nr,nc = x.shape
    xmean = np.zeros((nr,nc))
    x2    = np.zeros((nr,nc))
    # Metropolis algorithm
    for t in range(N):
        x=transition_all_TV(x,alpha, deltaTVL2A_pointwise, **opts)
        # update the mean
        if t >= t_burnin:
            tb = t - t_burnin
            xmean = tb/(tb+1)*xmean + 1/(tb+1)*x
            x2    = tb/(tb+1)*x2 + 1/(tb+1)*x**2
    stdx = np.sqrt(x2 - xmean**2)
    return x,xmean,stdx 