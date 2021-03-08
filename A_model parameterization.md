# A. Parameterization

<img src=".\param_comparison_2.png" alt="param_comparison_2" style="zoom:7%;" />

**Figure A1. Parameter values for wild-type and mutant simulations.**

Parameter names follow those of the four-state model as described in Chen et al. (2007). Briefly, 

$\alpha(V)$ and $$\beta(V)$$ are the forward and reverse rates for the resting-activated transition $C_R \leftrightarrow C_A$. Similarly, $\alpha'(V)$ and $\beta'(V)$ are the forward and reverse rates for $O_R \leftrightarrow O_A$. Each of these voltage-dependent transitions are given by:

$$ \alpha(V) = \alpha_0 \exp(-V/s_\alpha) \\ \beta(V) = \beta_0 \exp(V/s_\alpha) $$

For the wild-type data, parameters for fits using a four-state model (WT_4) and a six-state model (WT_6a, WT_6b; Figure A2 below) are shown. The latter has been shown as two sets, because this model has two voltage-dependent steps each between closed states and between open states. 

<img src=".\image-20210307235651594.png" alt="image-20210307235651594" style="zoom:50%;" />

**Figure A2. Six-state model from Hummert et al. (2018).**

In this model, the voltage-dependent steps are: $k_1$ to $k_4$ and $k_9$ to $k_{12}$. The remaining rates, which correspond to open-close transitions, are voltage-independent. From Figure 2 of Hummert et al. (2018).

**A1. The 'lumping' of six-state model parameters for visualization in Figure A1**.

To enable comparison with parameters for fits using the four-state model, parameters of a fit using the six-state model were lumped and divided into two sets: WT_6a and WT_6b. First, the voltage-dependent rates in the six-state model follow the same form as those given above for the four-state model. However, the 'slope' factors $s_\alpha$, $s_\beta$, etc., were constrained to reduce the number of free parameters. That is, the model uses two slope factors $s_1$ and $s_2$, such that a given voltage-dependent rate is parameterized as:

$ k_i = k^{0}_i \exp (\pm V / s_1) $ for $i = 1, 2, 9, 10$, and 

$k_i = k^{0}_i exp(\pm V / s_2) $ for $i = 3, 4, 11, 12$.

Where the coefficients $k^{0}_i$ are analogous to $\alpha_0$ and $\beta_0$ in the equations above for the four-state model. Therefore, the slope value divides the voltage-dependent parameters into two sets. 

Secondly, in Figure A1, the rates of forward and reverse transitions in each set of voltage-dependent parameters are categorized as $\alpha$ and $\beta$ parameters, respectively. For example:

$k^{0}_1$ and $k^{0}_9$ are categorized as $\alpha_0$ and $\alpha'_0$, respectively, whereas $k^{0}_2$ and $k^{0}_{10}$ are categorized as $\beta_0$ and $\beta'_0$, respectively. To reiterate, these four rates all share the same slope value, $s_1$, and this is reflected in Figure A1. Thus, for WT_6a, $s_1 = s_{\alpha} = s_{\beta} = s_{\alpha'} = s_{\beta'}$. 

For no particular reason, WT_6a was also chosen to contain the voltage-independent parameters governing open-close transitions. The naming of these parameters is analogous to the Chen model:

$g = k_7, h = k_6, g' = k_5,$ and $h' = k_6$. In either resting or activated states, $g$ refers to opening while $h$ refers to closing. Rates lacking the prime refer to transitions between fully-activated states, while the prime signifies the corresponding rates of open-close transitions at rest. 

**A2. Detailed balance.**

The use of identical slope values in the six-state model allows the model to satisfy detailed balance at all non-zero voltages. Detailed balance at 0 mV can then be established by choosing any of the voltage-dependent 'coefficient' terms (such as $k^{0}_1$) or any of the voltage-independent parameters. Chen et al. (2007) proposed that, in the four-state model, using $\alpha'_0$ (along with $s_\alpha'$) to establish detailed balance gave the best fit to their data. Here, we found that, out of a handful of parameters, including $k^{0}_9$, $k^{0}_{11}$, and $k_5$, the latter ($k_5$) seemed to give the best fit. See Colquhoun et al. (2004) for more discussion on detailed balance in kinetic models. 





**References**

Chen, S., Wang, J., Zhou, L., George, M.S., Siegelbaum, S.A., 2007. Voltage sensor movement and cAMP binding allosterically regulate an inherently voltage-independent closed-open transition in HCN channels. J. Gen. Physiol. 129, 175–188. https://doi.org/10.1085/jgp.200609585

Colquhoun, D., Dowsland, K.A., Beato, M., Plested, A.J.R., 2004. How to Impose Microscopic Reversibility in Complex Reaction Mechanisms. Biophys J 86, 3510–3518. https://doi.org/10.1529/biophysj.103.038679

Hummert, S., Thon, S., Eick, T., Schmauder, R., Schulz, E., Benndorf, K., 2018. Activation gating in HCN2 channels. PLoS Comput. Biol. 14, e1006045. https://doi.org/10.1371/journal.pcbi.1006045
