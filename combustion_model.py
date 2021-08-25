from ipywidgets import interactive, interactive_output, Layout, HBox, VBox, interact
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.constants import R
from scipy.integrate import odeint
from scipy.optimize import least_squares

class CombustionWidgets:
    """
    The Combustion Widgets class is used to create widgets sliders for determining
    initial value guesses for solving the ODEs for Coke Combustion. 
    
    ---------------
    INPUTS
    range_sliders = (dict) for adjusting the ranges of the ipywidgets sliders.
    
        example: {'C' : [max, min, stepsize], 'E' : [max, min, stepsize], 'k' : [max, min, stepsize]}
    
    """

    def __init__(self, *, range_sliders=0):
        self.range_sliders  = range_sliders

        if self.range_sliders == 0:
            self.range_sliders = {
                'C' : [15.5, 0, 1e-2],
                'E' : [4, 0.1, 1e-1],
                'k' : [20, 1e-2, 1e-2]
            }

    def set_parameters(self, n):
        """
        Function that creates sets of ipywidget sliders used for making initial guesses
        to non-linear least squares.
        
        ------------
        INPUTS
        n : Integer - Number of coke carbon species wanted.
        ------------
        OUTPUTS
        kwargs : dictionary of ipywidget sliders
        """
        

        Cs_sliders = [widgets.FloatSlider(  # List of carbon content slider
                value=(self.range_sliders['C'][0] + self.range_sliders['C'][1])/2,
                min=self.range_sliders['C'][1],
                max=self.range_sliders['C'][0],
                step=self.range_sliders['C'][2],
                description='C%d pct' % i,
                disabled=False,
                continuous_update=True,
                orientation='horizontal',
                readout=True,
                readout_format='.3f',
            ) for i in range(n)]

        E_sliders = [widgets.FloatSlider(  # List of activation energy sliders
                value=(self.range_sliders['E'][0] + self.range_sliders['E'][1])/2,
                min=self.range_sliders['E'][1],
                max=self.range_sliders['E'][0],
                step=self.range_sliders['E'][2],
                description='E%d kj/mol' % i,
                disabled=False,
                continuous_update=True,
                orientation='horizontal',
                readout=True,
                readout_format='.2f', style={'description_width': 'initial'}
            ) for i in range(n)]

        k_sliders = [widgets.FloatSlider(  # List of rate constant sliders
                value=(self.range_sliders['k'][0] + self.range_sliders['k'][1])/2,
                min=self.range_sliders['k'][1],
                max=self.range_sliders['k'][0],
                step=self.range_sliders['k'][2],
                description='k%d 1/min 1/bar' % i,
                disabled=False,
                continuous_update=True,
                orientation='horizontal',
                readout=True,
                readout_format='.4f', style={'description_width': 'initial'}
            ) for i in range(n)]

        # The sliders are placed into dictionaries
        Cs = {'C{}'.format(i):slider for i, slider in enumerate(Cs_sliders)}
        ks = {'k{}'.format(i):slider for i, slider in enumerate(k_sliders)}
        Es = {'E{}'.format(i):slider for i, slider in enumerate(E_sliders)}
        
        # We place all the sliders in a dictionary = we have a dictionary of dictionaries
        Guess_Sliders = {**Cs, **ks, **Es}

        return Guess_Sliders

class CokeCombustion:
    """
    The CokeCombustion Class defines the combustion model used in all other simulations
    involving coke combustion. The Combustion class simulates ideal combustion in oxygen
    atmosphere based on experimental TGA/DTA data, Activation Energy (E0, E1, ..., Ei), 
    Rate Constant (k0, k1, ..., ki), Mass Percent (C0, C1, ..., Ci) where "i" refers to the
    ith combustable mass fraction and is a desticint chemical species.

    The class offers interactivity via iPywidgets for finding E, k, and C.
    
    The class uses the ideal re-parametrized Arrhenius model described by: 
    
    Keskitalo, T. J., K. J. T. Lipiäinen, and A. O. I. Krause. “Modelling of Carbon and Hydrogen Oxidation
    Kinetics of a Coked Ferrierite Catalyst.” Chemical Engineering Journal, A special thematic issue on catalyst
    deactivation, 120, no. 1 (July 1, 2006): 63–71. https://doi.org/10.1016/j.cej.2006.03.033.

    ----------------------
    INPUTS
    initial_value_widgets : CombustionWidgets object defining the guess parameters for refinement.
    exp_data : A Pandas Dataframe. Importantly the following column names must be present in the dataframe:
                exp_data['Temp'] = Temperature data in Celsius
                exp_data['Loss'] = Percent Mass loss as a function of Temperature.
                exp_data['Dloss'] = Derivative of the Loss column.
    Po : Partial pressure of oxygen [Bar] (Defaults to 0.2 Bar).
    dTdt = Heat rate in C/min. (Defaults to 2 C/min)
    
    """
    def __init__(self, inital_value_widgets, exp_data, Po=0.2, dTdt=2):
        self.inital_value_widgets = inital_value_widgets
        self.Po = Po  # Oxygen partial pressure
        self.dTdt = dTdt  # Heating rate [C/min]
        self.exp_data = exp_data  # Exp. data
        self._temperatures =  exp_data['Temp'] #  Temperature data
        self.T_ref = np.round((np.sum(1/self._temperatures)/self._temperatures.shape[0])**-1, 0)  # Calculating weighted reference T.
        self.refined_parameters = []  # Container for refined parameters

    ##############################
    ## Coke Combustion Model
    ##############################

    def _dcidT(self, C, T, *param):
        """
        The coke combustion model for a single coke fraction. Re-parametrized Arrhenius equation.
        ------------
        INPUTS
        C : w% Coke
        T : Temperature [Celsius]
        *param : List of guess parameters i.e k, E, and reference temperature
        Po : Oxygen Partial pressue [Bar]
        dTdt : Heating Rate [K/min]
        -----------
        Units: 
        Ek :  Kj/mol
        k : 1/min 1/bar
        """

        k, Ek = list(param)[0]  # Assigning the guess parameters
        T_ref = self.T_ref + 273.15  #  Reference temperature in Kelvin
        T = T + 273.15  # Temperature in Kelvin

        
        dcidT = -(k)*np.exp(-(Ek*1e3)/R*(1/T-1/(T_ref)))*(C)*self.Po*self.dTdt**-1  # Coke model for a single coke species.
        return dcidT


    def _dCdT(self, coke_fr, T, *param):
        """
        Wrapper function over dcdT() that computes the coke combustion for several coke fractions.
        """
        m = len(coke_fr)  # Number of coke fractions
        dCsdT = np.array([self._dcidT(coke_fr[i], T, list(param)[0][i::m]) for i in range(m)])  #  Array of calculated coke derivatives
        return dCsdT

    
    def calc_combustion(self, **inital_values):
        """
        The calc_combustion function is the main function being called for computing
        coke combustion. It sorts the inital values and ODE parameters based on the input 
        ipywidgets. 
        """

        n = int(len(inital_values)/3)  # Number of coke frations
    
        Cs = list(inital_values.keys())[0:n]  # The coke fraction dictionary key names
        C = [inital_values.get(key) for key in Cs]  #  The initial coke content guess value
        parameters = [inital_values.get(key) for key in list(inital_values.keys())[n:]]  # Initial guess values for the parameters k, E.

    
        coke_loss = odeint(self._dCdT, C, self._temperatures, args=(parameters,)) # Integrates the combustion ODE with Scipy Integrate
        derivatives = -1*np.array([self._dcidT(coke_loss[:, i], self._temperatures, parameters[i::n]) for i in range(coke_loss.shape[1])]).T # Computes cokes loss derivatives
        mass_loss_sum = self._convert_to_loss(coke_loss, self.exp_data['Loss'])  # Normalizes coke loss to exp. data

        return coke_loss, self._temperatures, mass_loss_sum, derivatives

    def _convert_to_loss(self, sim_loss, exp_loss):
        """
        Normalizes computed coke loss to experimental data loss.
        """
        sim_sum = exp_loss.max() - (sim_loss.sum(axis=1).max() - sim_loss.sum(axis=1))

        return sim_sum

    @staticmethod
    def widgets_to_df(widget_values):
        """
        A simple static method to convert initial guess widgets to a pandas DataFrame
        """
        widget_value_dict = {val.description: val.value for val in widget_values}
        return pd.DataFrame(widget_value_dict, index=[0])



class TGACombustion(CokeCombustion):
    """
    The TGACombustion Subclass extends CokeCombustion with a non-linear least squares solver to
    to find the chemical parameters defined in the CokeCombustion parent class. 
    
    The use of a TGACombustion subclass is done in order allow for coupling the CokeCombustion to future Subclasses.
    """


    refined_parameters = None

    def __init__(self, inital_value_widgets, *, exp_data, Po=0.2, dTdt=2):
        super().__init__(inital_value_widgets, exp_data, Po, dTdt)

    ###############################
    ## Inital Guess with Widgets
    ###############################

    def __interactive_combustion(self, *, coke_fractions):
        
        # Below we create inital axis objects prior to widgets.interactive. This significantly speeds up 
        # the interactive plotting.

        plot_widget = widgets.Output()
        with plot_widget:
            fig, axs = plt.subplots(1, 2, figsize=[12, 5])
            axs[0].set_xlabel(r'Temperature ($^{\circ}$C)')
            axs[0].set_ylabel(r'DTA')
            axs[1].set_xlabel(r'Temperature ($^{\circ}$C)')
            axs[1].set_ylabel(r'Mass Loss (w/w %)')

        ax1, ax2 = axs

        derv_ax_objects = []
        for i in range(coke_fractions):
            derv_ax_objects.append(ax1.plot([], [])[0])

        if coke_fractions > 1:
            derv_ax_objects.append(ax1.plot([], [], ls='-.', label='Sim. Sum')[0])

        DTG_exp_plot, = ax1.plot(self.exp_data['Temp'], self.exp_data['Dloss'], ls='--', label='Exp.')

        Sim_Loss_Plot, = ax2.plot([], [], label='Sim.' )
        Loss_Exp_Plot, = ax2.plot(self.exp_data['Temp'], self.exp_data['Loss'], ls='--', label='Exp.')

        fig.canvas.header_visible = False

        #-----ipywidgets interactive function
        sim_result = interactive(
            self.__calc_and_plot_combustion,
            derv_ax=widgets.fixed(derv_ax_objects),
            loss_ax=widgets.fixed(Sim_Loss_Plot),
            **self.inital_value_widgets.set_parameters(coke_fractions)
            )
        #-------------------------------------

        self.guess_parameters = sim_result.children[:-1]

        labels = [val.description for val in self.guess_parameters][:self.coke_fractions] # Creating legend labels for the plots
        ax1.legend(labels)


        # Structuring of the plot layout
        cs = HBox(sim_result.children[:coke_fractions], layout = Layout(flex_flow='row wrap'))
        ks = HBox(sim_result.children[coke_fractions:2*coke_fractions], layout = Layout(flex_flow='row wrap'))
        Es = HBox(sim_result.children[2*coke_fractions:3*coke_fractions], layout = Layout(flex_flow='row wrap'))
        controls = HBox([cs, ks, Es])

        box_layout = Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    width='100%')


        main = VBox([plot_widget, controls], layout=box_layout)

        return display(main)

    def __calc_and_plot_combustion(self, derv_ax, loss_ax, **inital_values):
        
        coke_loss, temperatures, mass_loss_sum, derivatives = super().calc_combustion(**inital_values)  # We run the coke combustion calculation from the parent class

        #---------Set data for axis objects defined in "__simulate_combustion()"
        if self.coke_fractions > 1:
            derv_ax[-1].set_data(temperatures, derivatives.sum(axis=1))
            

        for i in range(self.coke_fractions):
            derv_ax[i].set_data(temperatures, derivatives[:, i])
            

        loss_ax.set_data(temperatures, mass_loss_sum)
        fig.draw()  # Draws the new data on the figure
        
        return 


    def simulate_TGA(self, *, res=[]):

        intslider = widgets.IntSlider(min=1, max=4, description='No. Coke Fractions', style={'description_width': 'initial'},
                                      layout=widgets.Layout(display='flex', flex_flow='center', align_items='center'))

        if len(res) == 0:
            @interact()
            def run(x=intslider):
                self.coke_fractions = x
                main = self.__interactive_combustion(coke_fractions=x)
                return main

    def yield_guess_parameters(self, guess_parameters):
        return np.array([val.value for val in guess_parameters])

    #######################################################
    ## LEAST SQUARES 
    #######################################################

    def least_squares_fit(self, *, parameters=[], coke_frac=0):
        """
        Least squares wrapper function over scipy.least_squares. Optimizes the
        coke combustion parameters. Also calculates the Rsq.
        """
        if (coke_frac and len(parameters)) != 0:
            self.guess_parameters = parameters
            self.coke_fractions = coke_frac
            function_parameters = self.guess_parameters
        else:
            function_parameters = self.yield_guess_parameters(self.guess_parameters)

        def rsq_calc(exp, residuals):
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((exp-np.mean(exp))**2)
            r_sq = 1 - (ss_res/ss_tot)
            return r_sq
        
        result = least_squares(self.fit_model, function_parameters, method='lm')  # scipy.least_squares()

        self.refined_parameters = result.x

        self.rsq = rsq_calc(self.exp_data['Dloss'], result.fun)

        return self

    def fit_model(self, function_parameters):
        """
        Wrapper function that goes into "least_square_fit()". It computes
        the coke combustion and returns the least square residuals, as required
        by scipy.least_squares().
        """
        coke_loss, mass_loss, derv = self.compute_TGA(function_parameters)
        derivatives_sum = derv.sum(axis=1)

        return self.exp_data['Dloss'] - derivatives_sum  # Residual array

    def compute_TGA(self, function_parameters):
        """
        Main combustion calculation function. Analogous to the parent class "calc_combustion()"
        except this is not working with widget valued parameters.
        """
        Cs = function_parameters[:self.coke_fractions]  # Coke weight fractions
        parameters = function_parameters[self.coke_fractions:] # Rest of the combustion model parameters

        coke_loss = odeint(super()._dCdT, Cs, self._temperatures, args=(parameters,)) # Integrating combustion model
        derivatives = -1*np.array([super(TGACombustion, self)._dcidT(coke_loss[:, i], self._temperatures, parameters[i::self.coke_fractions]) for i in range(coke_loss.shape[1])]).T  # Computing derivatives
        mass_loss_sum = super()._convert_to_loss(coke_loss, self.exp_data['Loss'])  # Computing mass loss sum of coke fractions

        return coke_loss, mass_loss_sum, derivatives

    #######################################
    ##  Plotting Result
    ######################################

    def plot_results(self):

        def build_parameter_dataframe(refined_parameters, widgets):
            C = ['C{}'.format(i) for i in range(self.coke_fractions)]
            E = ['E{}'.format(i) for i in range(self.coke_fractions)]
            k = ['k{}'.format(i) for i in range(self.coke_fractions)]

            df_dict = {key: value for (key, value) in zip(C + k + E, refined_parameters) }
            return pd.DataFrame(df_dict, index=[0])


        self.loss, self.dloss = self.compute_TGA(self.refined_parameters)[1:]

        box_layout = Layout(display='flex',
                    flex_flow='column',
                    justify_content='center',
                    width='100%')

        result_output = widgets.Output()
        parameters_output = widgets.Output(layout=Layout(display='flex', justify_content='center'))

        with result_output:
            fig_res, axs_res = plt.subplots(1, 2, figsize=[12, 5])

        labels = ['C{}'.format(i) for i in range(self.coke_fractions)]
        for i in range(self.coke_fractions):
            axs_res[0].plot(self.exp_data['Temp'], self.dloss[:, i], label=labels[i])

        if self.dloss.shape[1] > 1:
            axs_res[0].plot(self.exp_data['Temp'], self.dloss.sum(axis=1), label='Sim. Sum')

        axs_res[0].plot(self.exp_data['Temp'], self.exp_data['Dloss'], ls='--', label='Exp')
        axs_res[0].set_ylabel(r'DTG')

        axs_res[1].plot(self.exp_data['Temp'], self.loss, label='Sim. TGA')
        axs_res[1].plot(self.exp_data['Temp'], self.exp_data['Loss'], ls='--', label='Exp. TGA')
        axs_res[1].set_ylabel(r'% mass')
        axs_res[1].text(
            self.exp_data['Temp'].max()*0.82,
            self.exp_data['Loss'].max()*0.98,
            'Rsq: '+'{rsq:.{digits}f}'.format(rsq=self.rsq, digits=5)
            )

        for ax in axs_res:
            ax.set_xlabel(r'Temperature ($^\circ$C)')
            ax.legend()

        # fig_res.canvas.toolbar_position = 'bottom'
        fig_res.canvas.header_visible = False

        with parameters_output:
            display(build_parameter_dataframe(self.refined_parameters, self.guess_parameters))


        Title = widgets.HTML(
                    value='<p style="font-size:20px" align="center">Refined Parameters</p>'
                    )


        Plot_Title = widgets.HTML(
                    value='<p style="font-size:30px" align="center">Refined TGA</p>',
                    )


        Result_main = VBox([Title, parameters_output, Plot_Title, result_output], layout=box_layout)

        return display(Result_main)
    
    def export_results(self):
        
        dloss_data = {'Dloss_Comp_C{}'.format(i): self.dloss[:, i] for i in range(self.coke_fractions)}
        loss_data = {'Loss_Comp': self.loss}

        Cs = self.refined_parameters[:self.coke_fractions]
        Cs = {'C{}_wpct'.format(i): Cs[i] for i in range(self.coke_fractions)}

        Ks = self.refined_parameters[self.coke_fractions:2*self.coke_fractions]
        Ks = {'k{}'.format(i): Ks[i] for i in range(self.coke_fractions)}

        Es = self.refined_parameters[2*self.coke_fractions::]
        Es = {'E{}'.format(i): Es[i] for i in range(self.coke_fractions)}
        
        df_sim_results = pd.concat([pd.DataFrame(dloss_data), 
                                    pd.DataFrame(loss_data),
                                    pd.DataFrame(Cs, index=[0]),
                                    pd.DataFrame(Es, index=[0]),
                                    pd.DataFrame(Ks, index=[0])], axis=1)  # Merging simulation results to a dataframe

        df_total = pd.concat([self.exp_data.reset_index(drop=True), df_sim_results], axis=1)  # Merging simulation results with exp data
    
        return df_total

    
class XRDCombustion(CokeCombustion):

    def __init__(self, inital_value_widgets, *, exp_data, Po=0.2, dTdt=2):
        super().__init__(inital_value_widgets, exp_data, Po, dTdt)

