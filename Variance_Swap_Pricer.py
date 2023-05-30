# pip install --user pythonnet
# pip install --user matplotlib
import clr
from os.path import join
import sys
from sys import exit
from System import AppDomain


#folder where EQF is:
eqf_path = r'C:\Users\antoi\AppData\Local\Apps\EQFinance\EQFAddin'

useToolbox = True 

# ==============================================================================
# Then link the needed eqf dll's
if useToolbox:
    AppDomain.CurrentDomain.SetData('APP_CONFIG_FILE', join(eqf_path,'EQF.exe.config'))
    for assembly in ['EQFInterfaces', 'EQFAPI', 'EQFToolbox']:
        sys.path.append(eqf_path)
        clr.AddReference(assembly)
else:
    AppDomain.CurrentDomain.SetData('APP_CONFIG_FILE', join(eqf_path,'pythonEQF.exe.config'))
    for assembly in ['EQFInterfaces', 'EQFAPI']:
        clr.AddReference(join(eqf_path, assembly))

if not useToolbox: exit(2)


# ==============================================================================
ALLOW_GUI = False # True seems to work!

# ==============================================================================

import matplotlib.pyplot as plt

from EQFInterfaces import API
from EQFToolbox import Toolbox
from EQFInterfaces import DivAttribute
from EQFInterfaces import DivModel
from EQFInterfaces import Exercise
from EQFInterfaces import IBorrowCost
from EQFInterfaces import IDataItem
from EQFInterfaces import IDividend
from EQFInterfaces import IDividends
from EQFInterfaces import IScenario
from EQFInterfaces import IStressRule
from EQFInterfaces import ISpot
from EQFInterfaces import IStock
from EQFInterfaces import IVolSurface
from EQFInterfaces import IYieldCurve
from EQFInterfaces import OptType
from EQFInterfaces import SettlementType
from EQFInterfaces import IForward
from EQFInterfaces.Extensions import IContextExtensions
from EQFInterfaces.Extensions import IContextSetExtensions
from EQFInterfaces.Extensions import IEQFSchemaExtensions
from EQFInterfaces.Extensions import IYieldCurveExtensions

from System import DateTime
from System import Double
from System import String
clr.AddReference('System.Collections')
from System.Collections.Generic import List
from System.Collections.Generic import SortedList
from System.Threading import ApartmentState
from System.Threading import Thread
from System.Threading import ThreadStart

import yfinance as yf
from math import log, exp, sqrt
import numpy as np
import pandas as pd



# ==============================================================================


def app_thread():
    ''' Price vanilla creating all data, plots value and greeks vs spot '''

    #if not API.Initialise('Python', ALLOW_GUI, ..., None):
    if not Toolbox.Initialise('Python', False, ALLOW_GUI, None, None):
        print('EQF Initialise failed - check logfile')
        API.Shutdown()
        exit(1)

    def _copy_single(data_store, item):
        ''' Copy a single item into the datastore given and return the cached item '''
        list = List[IDataItem]()
        list.Add(item)
        return data_store.Copy[IDataItem](list)[0]


    #ToDo : def graph_vol_surface()
    

    def graph_yield_curve(yield_curve, as_of, step):
        #compute spot rates
        x = np.arange(0,10800,step)
        dates = [API.ToDateTime("{}D".format(i), as_of, as_of, None) for i in x]
        PVfactors = [IYieldCurveExtensions.PVFactor(yield_curve, as_of, date) for date in dates]
        rates = -np.log(PVfactors)/(x/365)
        #plot spot ratew
        plt.figure()
        plt.subplot(111)
        plt.plot(x/365, rates, 'k')
        plt.title('Spot Yield Curve')
        plt.xlabel('Maturity in Years')
        plt.show()

    def spot_shift_scenario(context,vanilla):
        X = [x * 0.01 for x in range(40,161)]
        scenario_definition = List[IScenario]()

        for mult in X:
            rules = List[IStressRule]()
            rules.Add(API.NewStressRule(None, 'spot', 'mult', stock.Handle.Name, mult))
            scenario_definition.Add(API.NewScenario(f'spotMult{mult}', None, rules))

        scenario_set = API.NewScenarioSet(None, DEFAULT_DATETIME, None, False, 'Linear', False, scenario_definition, None, False)

        context_set = API.NewContextSet(None, False, context, scenario_set, None)

        scenario_results = IContextSetExtensions.CalculateInstrumentStress(context_set, vanilla, None, None, True)

        results = [scenario_result.HeadlineResults for scenario_result in scenario_results]

        return results

    
    



    # ==========================================================================
    #CREATE A CONTEXT

    
    DEFAULT_DATETIME = DateTime()
    as_of = DateTime.UtcNow.Date
    data_store = Toolbox.DataStore
    currency = 'USD'
    name = "MSFT"


    #import ticker data
    ticker = yf.Ticker(name)
    ticker_historical_divs = ticker.get_dividends()
    ticker = yf.Ticker(name)
    ticker_prices = ticker.history(period="1d")['Close']
    


    #STOCK
    handle = API.NewDataItemHandle(IStock, name)
    stock = API.NewStock(handle, DEFAULT_DATETIME, None, None, currency, None, None, 'Created in Python', None)
    #set it in cache so that the vanilla can find its underlying:
    stock = _copy_single(data_store, stock)



    #SPOT
    SPOT = ticker_prices[-1]
    handle = API.NewDataItemHandle(ISpot, name, String.Empty, as_of)
    spot = API.NewSpot(handle, DEFAULT_DATETIME, None, SPOT)
    #print('Spot = ',spot.Value)



    #BORROW COST
    handle = API.NewDataItemHandle(IBorrowCost, name, String.Empty, as_of)
    borrow_cost = API.NewBorrowCost(handle, DEFAULT_DATETIME, None, None, True, as_of)
    


    #DIVIDENDS
    g = 0.01 #dividend growth rate 
    
    #get historical div from yahoo 
    last_div = float(ticker_historical_divs[-1])
    last_ex_date = str(ticker_historical_divs[[-1]].index[0])
    LAST_EX_DATE = API.ToDateTime(last_ex_date, as_of, as_of, None)
    print('last_div : ',last_ex_date, last_div)

    #construct schedule of quarterly payements
    schedule = List[IDividend]()

    for i in range(1,8):
        ABS = last_div*(1+g)**i
        PROP = 0.0
        add_months = i*3
        EX_DATE = API.ToDateTime("{}M".format(str(add_months)), LAST_EX_DATE, as_of, None)
        PAY_DATE = EX_DATE
        dividend_point = API.NewDividend(ABS, DivAttribute.Ordinary, EX_DATE, PAY_DATE, PROP)
        schedule.Add(dividend_point)
        
    handle = API.NewDataItemHandle(IDividends, name, String.Empty, as_of)
    dividends = API.NewDividends(handle, DEFAULT_DATETIME, None, True, as_of, SPOT, schedule)
    #print('Next dividend : ', dividends.Schedule[0].Date, dividends.Schedule[0].Abs)


    
    #VOL_SURFACE
    #from nameof import nameof

    nearVol = 0.18
    farVol = 0.24
    revTime = 0.5
    skew = -0.1
    smile = 0.4

    handle = API.NewDataItemHandle(IVolSurface, 'msft')
    vol_surface = API.NewDefaultVolSurface(handle,DEFAULT_DATETIME,None,True,DivModel.BSVol,None,"EQF4C3",None,1,1,0,as_of,SPOT)
   
    globalCurveParams = vol_surface.GlobalCurveParams
    globalCurveParams[1] = skew
    globalCurveParams[2] = smile
    termParams = vol_surface.TermParams
    termParams[0] = nearVol 
    termParams[1] = farVol
    termParams[2] = revTime

    vol_surface = IEQFSchemaExtensions.DeepCopyWith[IVolSurface](vol_surface,"GlobalCurveParams", globalCurveParams, "TermParams", termParams)








    #YIELD CURVE
    pv_zeroCoupon = SortedList[DateTime,Double]()
    pv_zeroCoupon.Add(API.ToDateTime("30Y",as_of,as_of,None) , 0.5262)
    pv_zeroCoupon.Add(API.ToDateTime("20Y",as_of,as_of,None) , 0.67)
    pv_zeroCoupon.Add(API.ToDateTime("10Y",as_of,as_of,None) , 0.8590)
    pv_zeroCoupon.Add(API.ToDateTime("5Y",as_of,as_of,None) , 0.9565)
    pv_zeroCoupon.Add(API.ToDateTime("2Y",as_of,as_of,None) , 0.9968)
    pv_zeroCoupon.Add(API.ToDateTime("1Y",as_of,as_of,None) , 0.9995)
    pv_zeroCoupon.Add(API.ToDateTime("6M",as_of,as_of,None) , 0.9998)
    pv_zeroCoupon.Add(API.ToDateTime("3M",as_of,as_of,None) , 0.9999)
    
    handle = API.NewDataItemHandle(IYieldCurve, currency, String.Empty, as_of)
    yield_curve = API.NewYieldCurve(handle, DEFAULT_DATETIME, None, currency, pv_zeroCoupon, as_of)

    #print('1Y SPOT PV FACTOR is : ',yield_curve.Data[API.ToDateTime("1Y",as_of,as_of,None)])

    #graph_yield_curve(yield_curve,as_of, step = 30)


    #Fill a Context
    context = API.NewContext('PyContext', as_of, None, False, None, None, None)
    context.Set(spot)
    context.Set(dividends)
    context.Set(borrow_cost)
    context.Set(vol_surface)
    context.Set(yield_curve)    



    # ================================================================================================
    #Create a data for the variance swap replicating portofolio

    #Set replicating portfolio variables 
    t = input('Enter an expiration YYYYMMDD : ')
    annualisation = 252
    expiry = API.ToDateTime(str(t), as_of, as_of, None)
    life = int(str(expiry.Date - as_of.Date)[:3])/252
    rate = 0.0005 
    strikes = [x * 0.01 for x in range(40,161)]
    deltaK = strikes[1] - strikes[0]
    #ToDo : create a function where you input params of the variance and it returns the params needed for the replicating portfolio
    #ToDo : create a function to compute the best step and range for the strikes of the vanillas to pick up, according to the vol, skew, maturity...



    #Compute forward level for the given expiry
    i = 0
    while dividends.Schedule[i].Date <= expiry:

        if i == 0:
            fwdPVFactor = IYieldCurveExtensions.PVFactor(yield_curve, as_of, dividends.Schedule[i].Date)
            fwd_level = spot.Value*fwdPVFactor - dividends.Schedule[i].Abs

        date1 = dividends.Schedule[i].Date
        date0 = dividends.Schedule[i-1].Date
        PVFactor1 = IYieldCurveExtensions.PVFactor(yield_curve, as_of, date1)
        PVFactor0 = IYieldCurveExtensions.PVFactor(yield_curve, as_of, date0)
        fwdPVFactor = PVFactor1 / PVFactor0        
        fwd_level = fwd_level*fwdPVFactor - dividends.Schedule[i].Abs    #ToDo : put borrow cost in the context and account for it in this formula

        i += 1
    
    fwdPVFactor = IYieldCurveExtensions.PVFactor(yield_curve, as_of, expiry) / PVFactor1
    fwd_level = fwd_level*fwdPVFactor 

    



    #Create a set of Vanillas   
    calls = [API.NewVanilla(None, DEFAULT_DATETIME, None, 
                            None,
                            currency, 
                            None, 
                            None, 
                            'Created in Python', 
                            None, 
                            None, 
                            1.0, 
                            stock.Handle, 
                            fwd_level*strike,   #forward MoneyNess
                            0.0, 
                            Exercise.European, 
                            OptType.Call, 
                            expiry, 
                            False, 
                            DEFAULT_DATETIME, 
                            SettlementType.Cash, 
                            None)
                            for strike in strikes]

    puts = [API.NewVanilla(None, DEFAULT_DATETIME, None, 
                            None,
                            currency, 
                            None, 
                            None, 
                            'Created in Python', 
                            None, 
                            None, 
                            1.0, 
                            stock.Handle, 
                            fwd_level*strike,    #forward MoneyNess
                            0.0, 
                            Exercise.European, 
                            OptType.Put, 
                            expiry, 
                            False, 
                            DEFAULT_DATETIME, 
                            SettlementType.Cash, 
                            None)
                            for strike in strikes]
    
    #Price the options
    call_results = [IContextExtensions.CalculateInstrument(context, calls[i], None, None, True) for i in range(len(calls))]
    put_results =  [IContextExtensions.CalculateInstrument(context, puts[i], None, None, True) for i in range(len(puts))]

    call_prices = [call_results[i].HeadlineResults['Price'] for i in range(len(call_results))]
    put_prices = [put_results[i].HeadlineResults['Price'] for i in range(len(put_results))]




##################################################################################################
##################################################################################################
###############Variance Swap###############


    #Compute option prices as percentage of fwd level
    call_as_pct_of_fwd = [x/fwd_level for x in call_prices]
    put_as_pct_of_fwd = [x/fwd_level for x in put_prices]

    
    #Create the corresponding dataframe
    vanilla_set = pd.DataFrame({'Strike' : strikes,
                                'Call' : calls,
                                'Put' : puts,
                                'Call_Result' : call_results,
                                'Put_Result' : put_results,
                                'Call/Fwd' : call_as_pct_of_fwd,
                                'Put/Fwd' : put_as_pct_of_fwd 
                                })
    
    


    #Define the weights for each vanilla : 1/K^2  
    vanilla_set['Call_Weight'] = deltaK/(vanilla_set['Strike']**2)
    vanilla_set['Put_Weight'] = deltaK/(vanilla_set['Strike']**2)

    #Remove ITMF options
    vanilla_set.loc[vanilla_set.Strike < 1 ,'Call_Weight'] = 0
    vanilla_set.loc[vanilla_set.Strike > 1,'Put_Weight'] = 0

    #Adjust the weights for the ATMF options
    vanilla_set.loc[vanilla_set.Strike == 1, 'Call_Weight'] = vanilla_set.loc[vanilla_set.Strike == 1, 'Call_Weight'] / 2 
    vanilla_set.loc[vanilla_set.Strike == 1, 'Put_Weight'] = vanilla_set.loc[vanilla_set.Strike == 1, 'Put_Weight'] / 2 
    
    #ToDo : use a more accurate method to compute weights 
    #ToDo : determine the round number for each option to buy 



    #Compute the cost of the replicating portfolio
    vanilla_set['Contribution'] = vanilla_set['Call_Weight']*vanilla_set['Call/Fwd'] + vanilla_set['Put_Weight']*vanilla_set['Put/Fwd']
    costOfRP = vanilla_set['Contribution'].sum()
    fair_strike = costOfRP*(2/life)*exp(rate*life)
    print('Variance Swap fair strike : ', sqrt(fair_strike))

    #Save the dataframe
    vanilla_set.to_csv('vanilla_set.csv')

    
    #Graph cash Gamma
    puts_gamma = [[scenario_result['Gamma'] for scenario_result in spot_shift_scenario(context,vanilla)] for vanilla in vanilla_set.Put]
    weighted_puts_gamma = [[gamma[i]*weight for i in range(len(puts_gamma[0]))] for gamma, weight in [(puts_gamma[i],vanilla_set.Put_Weight[i]) for i in range(len(strikes))]]
    calls_gamma = [[scenario_result['Gamma'] for scenario_result in spot_shift_scenario(context,vanilla)] for vanilla in vanilla_set.Call]
    weighted_calls_gamma = [[gamma[i]*weight for i in range(len(calls_gamma[0]))] for gamma, weight in [(puts_gamma[i],vanilla_set.Call_Weight[i]) for i in range(len(strikes))]]
    
    weighted_puts_gamma_df = pd.DataFrame(weighted_puts_gamma, index = strikes)
    weighted_calls_gamma_df = pd.DataFrame(weighted_calls_gamma, index = strikes)


    gamma = weighted_puts_gamma_df.sum() + weighted_calls_gamma_df.sum()
    scenario_results = spot_shift_scenario(context,vanilla_set.Call[0])
    spot_levels = [scenario_result['Spot'] for scenario_result in scenario_results]
    Y = [gamma[i]*(spot_levels[i]**2)*(2/life)*0.01 for i in range(len(spot_levels))]
    plt.plot(spot_levels,Y)
    plt.xlabel('Spot Level')
    plt.title('Cash Gamma of Replicating Portfolio')
    plt.show()

    

    if ALLOW_GUI: API.EditUserSettings()

    API.Shutdown()

def main():
    if ALLOW_GUI:
        thread = Thread(ThreadStart(app_thread))
        thread.SetApartmentState(ApartmentState.STA)
        thread.Start()
        thread.Join()
    else:
        app_thread()

if __name__ == '__main__':
    main()
    exit(0)