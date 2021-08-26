# Interactive Chemical Combustion

## General info

This is an interactive python class for simulating ideal combustion
based on non-linear least squares optimization from experimental
TGA/DTA (Thermal Gravimetric Analysis) data. 

By modelling you decomposition data you can extract chemical information
such as the activiation energy, rate constant, and mass fraction. The 
chemical combustion models allows deconvoluting of DTA curves and
subsequent mass fraction determination, followed by chemical parameters
for each mass fraction.

Please see the example notebook on how to use the combustion class.

## How To

To start the interactive combustion analysis:
Here we use the __CombustionWidgets__ class to provide 
interactive widgets for the combustion parameters. The 
range of the widget sliders can be configured independantly
by passing a dictionary formated accordingly.

{'C' : [max, min, step]}

```
import combustion_model as cm

simulation = cm.TGACombustion(
                cm.CombustionWidgets(
                          range_sliders={'C' : [15, 0, 1e-1], 
                                         'E' : [5e2, 0.01, 1e-3],
                                         'k' : [1e1, 1e-3, 1e-3]}
                                        ),
                                         exp_data=df,
                                         dTdt=10
                  )
```

The experimental data is passed as a pandas dataframe. The dataframe needs
columns with the following names:

```
df.Dloss  # %Mass loss derivative
df.Loss   # %Mass loss 
df.Temp   # Temperatures
```
Once the simulation has been created it is possible to determine
the combustion paramters interactivly:

```
simulation.simulate_TGA()
```
Finally, the parameter refinement can be done based on your guess parameters:

```
simulation.least_squares_fit().plot_results()
```

To export the results as a pandas dataframe:

```
df_results = simulation.export_results()
```
![bild](https://user-images.githubusercontent.com/70808555/131010097-bab87c83-cc5d-44c5-8a2b-1852b739e2ca.png)

## Dependencies

The python class uses ipywidgets and JupyterLab/Jupyter Notebook.
This needs to be configured in your jupyter environment.

___

