# B. Modelling methods and results.

****

**B1. Data preparation.**

The experimental data used for modelling was obtained from Figures 4A and 4B from Ramentol et al. (2020) were digitized using WebPlotDigitizer (https://automeris.io/WebPlotDigitizer/). The data were digitized to an approximate frequency of 2 kHz. However, not all traces could be digitized. For instance, deactivating tail currents were often not digitized because they overlapped for multiple test voltages. 

After digitization, the data were normalized using computed G-V and F-V data, which were provided in the Supplementary Information of the original publication. The lowest and highest values of opening and activation from the G-V and F-V data, respectively, were then used to normalize the digitized time courses. A similar procedure was previously described in Chen et al. (2007. 

**B2. Modelling methods.**

A variety of models were implemented by specifying their transition matrices and related information in separate scripts, and then called using a main script containing simulation/optimization methods. Model calibration (parameter fitting) was done entirely using scripts written in Julia 1.4.0 (https://julialang.org/). 

**B2a. Numerical approach.**

All kinetic models used follow the chemical 'master' equation: $\frac{dx}{dt} = Ax$, where $x$ is a vector of states and $A$ denotes the transition matrix. The elements of $A$ are defined as: 

- $A_{ij} = k_{ij}$ for $i \neq j$, and 
- $A_{ij} = -\sum^{N}_{j = 1 \\ i \neq j} k_{ij}$ for $i = j$, where $N$ is the total number of states.

We also have that $\sum x = 1$, so the number of equations can be reduced by one, yielding a modified form of the 'master' equation: $\frac{dx}{dt} = A_{N-1} x_{N-1} + k$, where the subscript $N-1$ indicates that the reduced dimensionality of the respective quantities. $k$ is a vector containing rates from the state which was removed. 

Numerical solution of the reduced system was done using a stiff solver (TR-BDF2) as implemented in the DifferentialEquations.jl (https://diffeq.sciml.ai/v2.1/) package. Absolute and relative tolerances were set to 1e-8 to avoid settling into local optima that can be observed when using low tolerances (Clerx et al., 2019). 

**B2b. Optimization.**

Model calibration was done using the implementation of Covariance Matrix Adaptive Evolution Strategy (CMAES) in GCMAES.jl (https://github.com/AStupidBear/GCMAES.jl). While ordinary CMAES algorithms are derivative-free, GCMAES.jl allows for optional specification of gradients. Thus, the gradient of a loss function was computed by autodifferentiation (https://github.com/JuliaDiff/ForwardDiff.jl) and used during optimization. 

A least-squares loss function was used in optimization, $Loss = Loss_I + Loss_F$, where the subscripts $I$ and $F$ refer to separate loss functions for the current and fluorescence traces. Each have the same form:

$Loss_x = f_x \frac{\sum(x - \hat{x})^2}{N} $,

where $x$ and $\hat{x}$ denote the data and model predictions, respectively. $f_x$ is a factor that was sometimes adjusted to alter the weight assigned to current versus fluorescence data. For instance, due to limitations in data digitization, the wild-type data contained more time courses for fluorescence than for current. Thus, $f_x$ were used to place more weight on the current data and lower that on the fluorescence data. 





**References**

Chen, S., Wang, J., Zhou, L., George, M.S., Siegelbaum, S.A., 2007. Voltage sensor movement and cAMP binding allosterically regulate an inherently voltage-independent closed-open transition in HCN channels. J. Gen. Physiol. 129, 175–188. https://doi.org/10.1085/jgp.200609585

Clerx, M., Beattie, K.A., Gavaghan, D.J., Mirams, G.R., 2019. Four Ways to Fit an Ion Channel Model. Biophysical Journal 117, 2420–2437. https://doi.org/10.1016/j.bpj.2019.08.001

Ramentol, R., Perez, M.E., Larsson, H.P., 2020. Gating mechanism of hyperpolarization-activated HCN pacemaker channels. Nat Commun 11, 1419. https://doi.org/10.1038/s41467-020-15233-9