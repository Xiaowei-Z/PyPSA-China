import pypsa, pandas as pd, numpy as np

from pypsa.descriptors import Dict

from pyomo.environ import Constraint


override_components = pypsa.components.components.copy()
override_components.loc["ShadowPrice"] = ["shadow_prices","Shadow price for a global constraint.",np.nan]
override_components.loc["CHP"] = ["chps","Combined heat and power plant.",np.nan]


override_component_attrs = Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["ShadowPrice"] = pd.DataFrame(columns = ["type","unit","default","description","status"])
override_component_attrs["ShadowPrice"].loc["name"] = ["string","n/a","n/a","Unique name","Input (required)"]
override_component_attrs["ShadowPrice"].loc["value"] = ["float","n/a",0.,"shadow value","Output"]

override_component_attrs["CHP"] = pd.DataFrame(columns = ["type","unit","default","description","status"])
override_component_attrs["CHP"].loc["name"] = ["string","n/a","n/a","Unique name","Input (required)"]
override_component_attrs["CHP"].loc["bus_fuel"] = ["string","n/a","n/a","Name of bus where fuel source is.","Input (required)"]
override_component_attrs["CHP"].loc["bus_elec"] = ["string","n/a","n/a","Name of bus where electricity is supplied.","Input (required)"]
override_component_attrs["CHP"].loc["bus_heat"] = ["string","n/a","n/a","Name of bus where heat is supplied.","Input (required)"]
override_component_attrs["CHP"].loc["p_nom_extendable"] = ["boolean","n/a",False,"","Input (optional)"]
override_component_attrs["CHP"].loc["capital_cost"] = ["float","EUR/MW",0.,"Capital cost per rating of electricity output.","Input (optional)"]
override_component_attrs["CHP"].loc["eta_elec"] = ["float","n/a",1.,"Electrical efficiency with no heat output, i.e. in condensing mode","Input (optional)"]
override_component_attrs["CHP"].loc["c_v"] = ["float","n/a",1.,"Loss of fuel for each addition of heat","Input (optional)"]
override_component_attrs["CHP"].loc["c_m"] = ["float","n/a",1.,"Backpressure ratio","Input (optional)"]
override_component_attrs["CHP"].loc["p_nom_ratio"] = ["float","n/a",1.,"Ratio of max heat output to max electrical output; max heat of 500 MWth and max electricity of 1000 MWth means p_nom_ratio is 0.5","Input (optional)"]
override_component_attrs["CHP"].loc["p_nom"] = ["float","MW",0.,"Nominal power for limits in OPF","Input (optional)"]
override_component_attrs["CHP"].loc["p_nom_max"] = ["float","MW","inf","If p_nom is extendable in OPF, set its maximum value (e.g. limited by technical potential).","Input (optional)"]
override_component_attrs["CHP"].loc["p_nom_min"] = ["float","MW",0.,"If p_nom is extendable in OPF, set its minimum value.","Input (optional)"]

override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]



class Network(pypsa.Network):

    def __init__(self,*args,**kwargs):
        kwargs["override_components"]=override_components
        kwargs["override_component_attrs"]=override_component_attrs
        super().__init__(*args,**kwargs)


    def lopf(self,*args,**kwargs):

        #at this point check that all the extra links are in place for the CHPs
        if not self.chps.empty:

            self.madd("Link",
                      self.chps.index + " electric",
                      bus0=self.chps.bus_source.values,
                      bus1=self.chps.bus_elec.values,
                      p_nom_extendable=self.chps.p_nom_extendable.values,
                      p_nom = self.chps.p_nom.values,
                      capital_cost=self.chps.capital_cost.values*self.chps.eta_elec.values,
                      efficiency=self.chps.eta_elec.values)

            self.madd("Link",
                      self.chps.index + " heat",
                      bus0=self.chps.bus_source.values,
                      bus1=self.chps.bus_heat.values,
                      p_nom_extendable=True,
                      efficiency=self.chps.eta_elec.values/self.chps.c_v.values)
        


        print("before solving, kwargs: ", kwargs)
        if "extra_functionality" in kwargs:
            user_extra_func = kwargs.pop('extra_functionality')
        else:
            user_extra_func = None

        #the following function should add to any extra_functionality in kwargs
        def extra_func(network, snapshots):
            #at this point add the constraints for the CHPs
            if not network.chps.empty:
                print("Setting up CHPs:",network.chps.index)

                print(network.links.p_nom)

                def chp_nom(model, chp):
                    return network.chps.at[chp,"eta_elec"]*network.chps.at[chp,'p_nom_ratio']*model.link_p_nom[chp + " electric"] == network.chps.at[chp,"eta_elec"]/network.chps.at[chp,"c_v"]*model.link_p_nom[chp + " heat"]

                network.model.chp_nom = Constraint(list(network.chps.index),rule=chp_nom)



                def backpressure(model,chp,snapshot):
                    return network.chps.at[chp,'c_m']*network.chps.at[chp,"eta_elec"]/network.chps.at[chp,"c_v"]*model.link_p[chp + " heat",snapshot] <= network.chps.at[chp,"eta_elec"]*model.link_p[chp + " electric",snapshot]

                network.model.backpressure = Constraint(list(network.chps.index),list(snapshots),rule=backpressure)



                def top_iso_fuel_line(model,chp,snapshot):
                    return model.link_p[chp + " heat",snapshot] + model.link_p[chp + " electric",snapshot] <= model.link_p_nom[chp + " electric"]

                network.model.top_iso_fuel_line = Constraint(list(network.chps.index),list(snapshots),rule=top_iso_fuel_line)



            if user_extra_func is not None:
                print("Now doing user defined extra functionality")
                user_extra_func(network,snapshots)
                print("Done with user defined extra functionality")

        kwargs["extra_functionality"]=extra_func
        print("kwargs passsed to super().lopf() has: ", kwargs)

        super().lopf(*args,**kwargs)
