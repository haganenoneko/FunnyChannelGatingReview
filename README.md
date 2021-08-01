# FunnyChannelGatingReview
Update (July 22, 2021): Code will be uploaded shortly (within a week or so) after resolving some logistic issues. 

Supplementary figures and code for the review "Analysis of voltage-dependent gating of funny channels by kinetic models"

## Data files
We provide normalized time courses and steady-state data in .CSV files, compressed in `data.rar`, which can be opened using [WinRAR](https://www.win-rar.com/start.html?&L=0)
```
├── data.rar
│   ├── ramentol_normalized_wt.png    # plot of normalized WT current and fluorescence time traces 
│   ├── wt_ft_act.csv                 # WT fluorescence, activation 
│   ├── wt_ft_de.csv                  # WT fluorescence, deactivation
│   ├── wt_it_act.csv                 # WT current, activation
│   ├── wt_it_de.csv                  # WT current, deactivation
│   ├── wt_ss.csv                     # WT steady-state fluorescence and conductance
│   ├── ramentol_normalized_mut.png    # plot of normalized WT current and fluorescence time traces 
│   ├── mut_ft_act.csv                 # Mutant fluorescence, activation 
│   ├── mut_ft_de.csv                  # Mutant fluorescence, deactivation
│   ├── mut_it_act.csv                 # Mutant current, activation
│   ├── mut_it_de.csv                  # Mutant current, deactivation
│   ├── mut_ss.csv                      # Mutant steady-state fluorescence and conductance
```

## Scripts
`main_v4.jl` performs model-fitting using differential evolution or Covariance Matrix Adaptive Evolution Strategy (CMAES). Directories to data (above) and model (below) files need to be changed according to where the corresponding files are located. 

`four_3s_ramentol.jl` and `six_hum.jl` specify the kinetic models (four- and six-state, respsectively). 

Note that the implementations of simulations and fitting are not well-optimized, although common performance tips are suggested in each file. 

The scripts were written in `Julia 1.4.0`, but are compatible with newer versions, as well (`1.6.1` at the time of writing). Optimization takes about 2-4 hours on a laptop with the following specifications
```
> versioninfo()
Platform Info:
  OS: Windows (x86_64-w64-mingw32)
  CPU: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, skylake)
```

## Miscellaneous
'additional fitting results for wt and mutant with four state model.docx' shows simulations for the wild-type and depolarization-activated mutant under different conditions, such as without detailed balance, fitting only the activating step of the protocol, or the conditions shown in Figure 2 of the main text. 

## Contact
Questions can be directed to either Delbert Yip at d.yip@alumni.ubc.ca, or Eric Accili at eaccili@mail.ubc.ca
