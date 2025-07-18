from typing import Union
import os
import pandas as pd
import vectorbt as vbt
import numpy as np
import plotly.graph_objects as go
from typing import Literal

fdel = os.path.sep
wd = os.path.dirname(__file__)  ## This gets the working directory which is the folder where you have placed this .py file. 
# Get total PnL for each FY

    # Add each portfolio's value curve
    # portfolios = {
    #     'HoDlEr: Buy & Hold (bought on 2020-03-01)': bh_pf,
    #     'Max trades': inout_ords,
    #     'Long green, sell yellow/red': less_ords,
    #     'VAMS signal dependent sizing': cpf_fo
    # }

def plot_portfolios(portfolios: dict, return_fig: bool = False, title: str = "Portfolio Value comparison", 
                    annotation: str = " ", ann_box_pos: tuple = (0.4, 0.98)):
    """Compare the value curves of multiple portfolios.

    **Parameters:**
    - portfolios: dict - A dictionary of portfolio objects to compare. The keys are the names of the portfolios and the values are the portfolio objects.
    - title: str - The title of the plot.
    - annotation: str - The subtitle of the plot.
    """
    # Create figure
    fig = go.Figure()
    # Add subtitle by creating a figure with layout that includes both title and subtitle
    for name, pf in portfolios.items():
        if isinstance(pf, vbt.Portfolio):
            x = pf.value().index
            y = np.round(pf.value().values, 2)  # Round to 2 decimal places
        elif isinstance(pf, pd.Series):
            x = pf.index
            y = np.round(pf.values, 2)  # Round to 2 decimal places
        else:
            print(f"Unknown portfolio type: {type(pf)}")
            continue

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode='lines',
                line=dict(width=1.5)
            ))

    # Update layout with horizontal legend at bottom
    fig.update_layout(
        title=title,
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        width=1100,
        height=600,
        yaxis_type='log',
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        )
    )

    if return_fig:
        return fig
    else:
        fig.show()

def plot_timeseries_with_vlines(price_series, vline_dict=None, title='Time Series with Event Lines', 
                              width=1000, height=500, y_axis_title='Value'):
    """Plot a time series with vertical lines at specified dates using Plotly.
    
    Parameters:
    -----------
    price_series : pd.Series
        Time series data to plot with datetime index
    vline_dict : dict, optional
        Dictionary with keys as color names and values as pd.Series with datetime indices
        Example: {'red': pd.Series(index=[dt1, dt2]), 'green': pd.Series(index=[dt3, dt4])}
    title : str, optional
        Title for the plot
    width : int, optional
        Width of the plot in pixels
    height : int, optional
        Height of the plot in pixels
    y_axis_title : str, optional
        Title for the y-axis
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    # Create the main figure
    fig = go.Figure()
    
    # Add the main time series line
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            mode='lines',
            name=price_series.name if hasattr(price_series, 'name') else 'Series',
            line=dict(width=2)
        )
    )
    
    # Add vertical lines if provided
    if vline_dict is not None:
        for color, date_series in vline_dict.items():
            # Get datetime indices
            dates = date_series.to_list()
            
            # Add vertical lines for each date
            for date in dates:
                fig.add_shape(
                    type="line",
                    x0=date,
                    x1=date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color=color,
                        width=1.5,
                        dash="dash",
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title=y_axis_title,
        template='plotly_white',
        width=width,
        height=height,
        showlegend=True,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

# Create CPI Deflator Index for Real Value Conversion

def create_cpi_deflator(cpi_data: pd.Series, base_date: str = "2018-03-31")-> pd.Series:
    """
    Create a CPI deflator index for converting nominal values to real values.
    
    Parameters:
    -----------
    cpi_data : pd.Series
        CPI index values with datetime index
    base_year : int
        Base year for the deflator (default: 2018)
        
    Returns:
    --------
    pd.Series
        Deflator index where base_year = 1.0
    """

    # Find the closest date to the specified base_date
        # Convert the Datestring to a Timestamp object
    date_ts = pd.to_datetime(base_date)
    
    # Ensure all elements in index are Timestamp objects
    try:
        index = pd.to_datetime(cpi_data.index)
    except Exception as e:
        print(f"Error converting index to datetime: {e}")
        return None

    # Check for any non-datetime values in the index
    if not all(isinstance(x, pd.Timestamp) for x in index):
        print("Index contains non-datetime values.")
        return None
    
    # Find the closest date in the index
    closest_date = min(index, key=lambda x: abs((x - date_ts).total_seconds()))
    index_loc = index.get_loc(closest_date)

    base_cpi = cpi_data.iloc[index_loc]
    print(f"Using closest available date {closest_date} to requested base date {base_date}")
    # Create deflator: base year = 1.0, other years scale accordingly
    deflator = base_cpi / cpi_data
    deflator = pd.Series(deflator.values, index=pd.DatetimeIndex(deflator.index))
    print(f"CPI Deflator created with base date {base_date}")
    print(f"Base year CPI value: {base_cpi:.2f}")
    print(f"Deflator range: {deflator.min():.3f} to {deflator.max():.3f}")
    
    return deflator

def deflate_series(nominal_series: pd.Series, deflator: pd.Series, base_year=2018)-> pd.Series:
    """
    Convert nominal values to real values using CPI deflator.
    
    Parameters:
    -----------
    nominal_series : pd.Series
        Nominal values to be deflated
    deflator : pd.Series
        CPI deflator index
    base_year : int
        Base year for reference
        
    Returns:
    --------
    pd.Series
        Real values in constant base year dollars
    """
    
    # Align the series by reindexing deflator to match nominal_series dates
    aligned_deflator = deflator.reindex(nominal_series.index, method='ffill')
    
    # Convert to real values
    real_series = nominal_series * aligned_deflator
    
    real_series.name = f"{nominal_series.name}_real_{base_year}" if nominal_series.name else f"real_{base_year}"
    
    return real_series

#FUNCTIONS #########################################################################################################################
def defaultTaxRates() -> tuple:
    print("Using default tax information for Australia from 2019 - 2025.")
    taxRates = pd.read_excel(wd+fdel+"tax_rates_aus.xlsx", index_col=0)
    
    # Convert integer year index to DatetimeIndex with June 30th dates (end of financial year)
    if not isinstance(taxRates.index, pd.DatetimeIndex):
        # Assume index contains years as integers (2014, 2015, etc.)
        taxRates.index = pd.to_datetime([str(year)+"-06-30" for year in taxRates.index])
    
    tax_brackets = {bracket: group for bracket, group in taxRates.groupby('Bracket')}

    return taxRates, tax_brackets

def determine_tax_bracket(gross_income: float, year: int, tax_rates_aus: pd.DataFrame = None, deductions: float = 0) -> str:
    """Determine the tax bracket for a given gross income and year.

    **Parameters:**
    - gross_income: float - The gross income for the financial year.
    - year: int - The financial year (e.g. 2021 for 2020-21).
    - tax_rates: pd.DataFrame - The tax rates information dataframe. If None, uses default Australian tax rates.
    - deductions: float - Total deductions in AUD.

    **Returns:**
    - str - The tax bracket for the given gross income and year.
    """
    if tax_rates_aus is None:
        tax_rates_aus = defaultTaxRates()[0] # Get the default tax rates table

    net_income = gross_income - deductions
    bracket_row = tax_rates_aus.loc[
        (tax_rates_aus.index == str(year)+"-06-30") & 
        (tax_rates_aus["Bracket minimum (threshold)"] <= net_income) &
        ((tax_rates_aus["Bracket maximum"].isna()) | (net_income <= tax_rates_aus["Bracket maximum"]))
    ]
    
    if not bracket_row.empty:
        return bracket_row.iloc[0]["Bracket"]
    else:
        return "Unknown"

def create_trades_df(pf: vbt.Portfolio, price: pd.Series, aud: pd.Series, signal: pd.Series, deferred_trade: pd.DataFrame = None) -> pd.DataFrame:
    """ Add more info to the trades dataframe from a Portfolio backtest object.

    **Parameters:**
    - pf: vbt.Portfolio - The portfolio object to get the trades from.
    - price: pd.Series - The price series used for the backtest.
    - aud: pd.Series - The USD/AUD price history for conversion of pnl to AUD.
    - signal: pd.Series - The signal series used for the backtest.

    **Returns:**
    - pd.DataFrame - The trades dataframe with more info.
    """
    # Check if we have any trades
    if len(pf.trades.records) == 0:
        return pd.DataFrame()  # Return empty dataframe if no trades

    trades = pf.trades.records.drop(columns=["col", "direction", "status", "parent_id"])
    trades['entry_date'] = price.index[trades['entry_idx']]
    trades["id"] = trades["entry_date"].apply(lambda x: f"{x.year}_")+trades.index.astype(str)
    trades['exit_date'] = price.index[trades['exit_idx']]
    trades["return"] *= 100
    trades["duration_(days)"] = (trades['exit_date'] - trades['entry_date']).dt.days
    trades.rename(columns={"return": "return_(%)", "pnl": "pnl_(USD)"}, inplace=True)
    trades["entry_signal"] = signal.loc[trades["entry_date"]].values
    trades["exit_signal"] = signal.loc[trades["exit_date"]].values
    trades["pay_cgt"] = trades["exit_signal"].apply(lambda x: "defer" if pd.isna(x) else "pay")
    
    # Debug information
    print(f"Number of trades: {len(trades)}")
    print(f"Price date range: {price.index[0]} to {price.index[-1]}")

    for i in range(len(trades)):
        trades.loc[i, "financial_year"] = int(trades.loc[i, "exit_date"].year) if trades.loc[i, "exit_date"].month < 7 else int(trades.loc[i, "exit_date"].year) + 1
    
    # Check if we have AUD data for all trade dates
    missing_dates = trades['exit_date'][~trades['exit_date'].isin(aud.index)]
    if len(missing_dates) > 0:
        print(f"Warning: Missing AUD rates for {len(missing_dates)} trade dates")
        print("First few missing dates:", missing_dates.head())
    
    # Using pandas asof to match dates (will use the most recent rate if exact match not found)
    trades["pnl_(AUD)"] = trades["pnl_(USD)"] * aud.reindex(trades['exit_date'], method='ffill').values
    trades['financial_year'] = trades['financial_year'].astype(int)

    reopened = trades[pd.isna(trades["entry_signal"])]
    #This below should work for a trade closed the year after it was opened but may fail for multi-year long trades
    if not reopened.empty and deferred_trade is not None:
        total_duration = reopened["duration_(days)"].sum() + deferred_trade.iloc[0]["duration_(days)"]
        print("Adding deferred trade to trades dataframe")
        trades.loc[trades.index[0], "duration_(days)"] = total_duration #Add the duration of the deferred trade to the reopened trade
        deferred_trade.loc[deferred_trade.index[0], "duration_(days)"] = total_duration
        deferred_trade.loc[deferred_trade.index[0], "pay_cgt"] = "pay" #Set the deferred trade to pay CGT
        print("Length of trades before adding deferred trade:", len(trades))
        trades = pd.concat([deferred_trade, trades], axis = 0)
        print("Length of trades after adding deferred trade:", len(trades))
    return trades

def tax_calc(trades: pd.DataFrame,
             income: Union[float, list], 
             deductions: Union[float, list] = 0, 
             tax_rates: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the tax for each financial year based on the trades and tax rates and income etc.

    **Parameters:**
    - trades: pd.DataFrame - This is the trades dataframe from a vbt.Portfolio backtest with the following columns:
        ['size', 'entry_idx', 'entry_price', 'entry_fees', 'exit_idx', 'exit_price', 'exit_fees', 'pnl_(USD)', 'return_(%)', 'entry_date',
       'exit_date', 'duration_(days)', 'pnl_(AUD)', 'financial_year', 'fy_cumulative_pnl', 'fy_cumulative_pnl(AUD)']
       - There will be a function to produce this dataframe from a vbt.Portfolio backtest. Still need to write it.....
    - income: float or pd.Series. If float then the same income is applied to all financial years. If pd.Series then the income is applied to the corresponding financial year.
    - deductions: float or pd.Series, optional default is None. If float then the same deductions are applied to all financial years. If pd.Series then the deductions are applied 
    to the corresponding financial year. If None then the deductions are set to 0.
    - tax_rates: pd.DataFrame - This is the tax rates information dataframe. Has columns: ["bracket", "tax_rate_(%)",	"base_tax_aud",	"bracket_min_aud",	"bracket_max_aud"] 
    and index is the financial year in format of YYYY-YY e.g "2020-21". We could later generalize this to work with non-Australian tax rates.

    **Returns:**
    - pd.DataFrame - Payable tax information
    """
        # Calculate the cgt_rate based on the duration of the trade, giving 50% discount for trades held longer than 12 months
    trades["cgt_rate"] = 1.0
    # Use vectorized operation to avoid SettingWithCopyWarning
    long_term_gains = (trades["duration_(days)"] > 365) & (trades["pnl_(AUD)"] > 0)
    trades.loc[long_term_gains, "cgt_rate"] = 0.5

    # Calculate the CGT amount in AUD
    trades["cgt_amount_(AUD)"] = trades["pnl_(AUD)"] * trades["cgt_rate"]

    ## The basic bits
    fy_totals = pd.Series(trades.groupby('financial_year')["cgt_amount_(AUD)"].sum().round(2)).to_frame()

    if tax_rates is None:
        tax_rates = defaultTaxRates()[0] # Get the default tax rates table

    fy_totals["trades"] = pd.Series(trades.groupby("financial_year").size().to_list(), index=fy_totals.index)
    fy_totals["gross_taxable_income_(AUD)"] = pd.Series(income, index = fy_totals.index) + fy_totals["cgt_amount_(AUD)"]
    fy_totals["gross_deductions_(AUD)"] = pd.Series(deductions, index = fy_totals.index)
    fy_totals["net_taxable_income_(AUD)"] = fy_totals["gross_taxable_income_(AUD)"] - fy_totals["gross_deductions_(AUD)"]

    #Figure out the applicable tax bracket for each financial year
    fy_totals["tax_bracket"] = fy_totals.apply(
    lambda row: tax_rates.loc[
        (tax_rates.index == row.name) &  # Match the financial year
        (tax_rates["bracket_min_aud"] <= row["net_taxable_income_(AUD)"]) &  # Income is above bracket minimum
        ((tax_rates["bracket_max_aud"].isna()) |  # Either no upper limit (for highest bracket)
         (row["net_taxable_income_(AUD)"] <= tax_rates["bracket_max_aud"]))]["bracket"].iloc[0], axis=1)
    
    # Figure out the payable tax for each financial year
    payable = []; taxcomps = []
    if not isinstance(income, list):
        inco = [income for i in range(len(fy_totals))]
        print(f"Income λιστ μαδε: {inco}")
    else:
        inco = income

    for i in range(len(fy_totals)):
        fy = fy_totals.index[i]
        row = tax_rates.loc[(tax_rates.index == fy) & (tax_rates["bracket"] == fy_totals.loc[fy, "tax_bracket"])]
        bt = row.loc[fy, "base_tax_aud"]
        rate = row.loc[fy, "tax_rate_(%)"]
        brackmin = row.loc[fy, "bracket_min_aud"]
        taxable = fy_totals.loc[fy, "net_taxable_income_(AUD)"]
        #print(f"FY: {fy}, taxable: {taxable}, bracket minimum: {brackmin}, rate: {rate}, bt: {bt}")
        tax = bt + (taxable - brackmin) * (rate/100)
        non_cgt = (inco[i]/taxable)*tax
        cgt = tax - non_cgt
        payable.append(tax)
        taxcomps.append((non_cgt, cgt))
    fy_totals["total_tax_aud"] = pd.Series(payable, index=fy_totals.index)
    fy_totals["taxed_(% of gross income)"] = (fy_totals["total_tax_aud"] / fy_totals["gross_taxable_income_(AUD)"])*100
    fy_totals["non_cgt_tax_(aud)"] = pd.Series([tax[0] for tax in taxcomps], index=fy_totals.index)
    fy_totals["cgt_tax_(aud)"] = pd.Series([tax[1] for tax in taxcomps], index=fy_totals.index)
    return fy_totals, trades

def single_year_tax(year: int, income: float, year_trades: pd.DataFrame, usdaud: float,
                    previous_cg_loss: float = 0, deductions: float = 0, tax_rates: pd.DataFrame = None,
                    constant_cgt_discount: float = None,
                    fixed_cgt_rate: float = None) -> pd.Series:
    """Calculate the tax to pay for a single financial year from CGT and non-CGT income.
    
    **Parameters:**
    - year: int - The financial year (e.g. 2021 for 2020-21) - the year is the year the tax is paid.
    - income: float - Non-CGT income in AUD
    - previous_cg_loss: float - Capital loss that can be carried forward from previous years.
    - year_trades: pd.DataFrame - Trades dataframe from a vbt.Portfolio backtest for that year.
    - usdaud: float - USD/AUD exchange rate to convert capital gains to AUD on the tax payment date.
    - deductions: float - Total tax deductions in AUD
    - tax_rates: pd.DataFrame - Tax rates information (defaults to Australian tax rates and will be loaded from file if not provided)
    
    **Returns:**
    - pd.Series - Tax information for the year with the following index:
    
        - cgt_amount_(AUD): Capital gains amount in AUD
        - gross_taxable_income_(AUD): Total taxable income
        - gross_deductions_(AUD): Total deductions
        - net_taxable_income_(AUD): Net taxable income after deductions
        - total_tax_aud: Total tax
        - taxed_(% of gross income): Tax as percentage of gross income
        - non_cgt_tax_(aud): Tax on non-CGT income
        - cgt_tax_(aud): Tax on capital gains
        - capital_loss_carryforward_(AUD): Amount of capital loss that can be carried forward to next year
    """

    if tax_rates is None:
        tax_rates = defaultTaxRates()[0]  # Get the default tax rates table

    # Calculate CGT amount from trades
    if year_trades.empty:
        cgt_aud = 0
        capital_loss_carryforward = previous_cg_loss
        deferred = pd.DataFrame()
    else:
        # Calculate the cgt_rate based on the duration of the trade, giving 50% discount for trades held longer than 12 months
        if constant_cgt_discount is not None:
            year_trades["cgt_rate"] = constant_cgt_discount
        else:
            year_trades["cgt_rate"] = 1.0
            # Use vectorized operation to avoid SettingWithCopyWarning
            long_term_gains = (year_trades["duration_(days)"] > 365) & (year_trades["pnl_(AUD)"] > 0)
            year_trades.loc[long_term_gains, "cgt_rate"] = 0.5
    
        # Calculate the CGT amount in AUD
        year_trades["cgt_amount_(AUD)"] = year_trades["pnl_(AUD)"] * year_trades["cgt_rate"]

        #Exclude trades that are not closed yet, these are labelled as "defer" in the trades dataframe
        year_toPay = year_trades[year_trades["pay_cgt"] == "pay"]
        deferred = year_trades[year_trades["pay_cgt"] == "defer"]

        cgt_aud = year_toPay["cgt_amount_(AUD)"].sum()
        deferred_cgt = deferred["cgt_amount_(AUD)"].sum()
        print(f"Closed trades this year: \n{year_toPay}\nDeferred trades: {deferred}")
        print(f"Tax calculation for year {year}, capital gains total: {cgt_aud}, Deferred capital gains: {deferred_cgt} AUD")
        # Store the original CGT amount for carryforward calculation
        capital_loss_carryforward = abs(cgt_aud) + previous_cg_loss if cgt_aud < 0 else 0
    
    print(f"Capital loss carryforward: {capital_loss_carryforward} AUD")
    # Handle negative CGT
    if cgt_aud < 0:
        # For negative CGT, set CGT-related values to 0
        cgt_tax = 0
        # Calculate tax only on non-CGT income
        gross_taxable_income = income
        net_taxable_income = gross_taxable_income - deductions
        
        # Find applicable tax bracket
        bracket_row = tax_rates.loc[
            (tax_rates.index == year) & 
            (tax_rates["bracket_min_aud"] <= net_taxable_income) &
            ((tax_rates["bracket_max_aud"].isna()) | (net_taxable_income <= tax_rates["bracket_max_aud"]))
        ].iloc[0]
        
        # Calculate tax
        base_tax = bracket_row["base_tax_aud"]
        tax_rate = bracket_row["tax_rate_(%)"]
        bracket_min = bracket_row["bracket_min_aud"]
        
        total_tax = base_tax + (net_taxable_income - bracket_min) * (tax_rate/100)
        non_cgt_tax = total_tax  # All tax is from non-CGT income
    else:
        # Calculate total taxable income
        deductions += previous_cg_loss
        gross_taxable_income = income + cgt_aud
        net_taxable_income = gross_taxable_income - deductions
        
        # Find applicable tax bracket
        bracket_row = tax_rates.loc[
            (tax_rates.index == year) & 
            (tax_rates["bracket_min_aud"] <= net_taxable_income) &
            ((tax_rates["bracket_max_aud"].isna()) | (net_taxable_income <= tax_rates["bracket_max_aud"]))
        ].iloc[0]
        
        # Calculate tax
        base_tax = bracket_row["base_tax_aud"]
        tax_rate = bracket_row["tax_rate_(%)"]
        bracket_min = bracket_row["bracket_min_aud"]
        
        total_tax = base_tax + (net_taxable_income - bracket_min) * (tax_rate/100)
        
        # Split tax between CGT and non-CGT income
        non_cgt_tax = (income/gross_taxable_income) * total_tax
        if fixed_cgt_rate is not None:  #Adds a variable to force a fixed CGT rate, could apply to super or for top tax bracket etc.
            cgt_tax = fixed_cgt_rate * cgt_aud
        else:
            cgt_tax = total_tax - non_cgt_tax
    
    # Create result series
    result = pd.Series({
        "closed_PnL_(AUD)": year_toPay["pnl_(AUD)"].sum(),
        "cgt_amount_(AUD)": cgt_aud,
        "salary_income_(AUD)": income,
        "gross_taxable_income_(AUD)": gross_taxable_income,
        "gross_deductions_(AUD)": deductions,
        "net_taxable_income_(AUD)": net_taxable_income,
        "tax_bracket": bracket_row["bracket"],
        "bracket_min_(aud)": bracket_row["bracket_min_aud"],
        "base_tax_(aud)": base_tax,
        "tax_rate_(%)": tax_rate,
        "total_tax_aud": total_tax,
        "taxed_(% of gross income)": (total_tax / gross_taxable_income) * 100,
        "non_cgt_tax_(aud)": non_cgt_tax,
        "cgt_tax_(aud)": cgt_tax,
        "non_cgt_tax_(% of total tax)": (non_cgt_tax / total_tax) * 100 if total_tax > 0 else 0,
        "cgt_tax_(% of total tax)": (cgt_tax / total_tax) * 100 if total_tax > 0 else 0,
        "cgt_tax_paid_(usd)": cgt_tax / usdaud,
        "capital_loss_carryforward_(AUD)": capital_loss_carryforward
    })
    
    # Change the status of the paid cgt trades to "paid"
    year_toPay.loc[:, "pay_cgt"] = f"paid_{year}"
    print(f"Trades to pay CGT for {year}: {year_toPay}")
    return result, deferred, year_toPay

def calculate_trade_statistics(trades_df):
    """
    Calculate trade statistics from a trades DataFrame.
    
    Parameters:
    -----------
    trades_df : pd.DataFrame
        DataFrame containing trade records with at least 'return_(%)' and 'duration_(days)' columns
        
    Returns:
    --------
    dict
        Dictionary with trade statistics
    """

    if trades_df.empty:
        return {
            'Win Rate [%]': float('nan'),
            'Best Trade [%]': float('nan'),
            'Worst Trade [%]': float('nan'),
            'Avg Winning Trade [%]': float('nan'),
            'Avg Losing Trade [%]': float('nan'),
            'Avg Winning Trade Duration': pd.NaT,
            'Avg Losing Trade Duration': pd.NaT,
            'Profit Factor': float('nan'),
            'Expectancy': float('nan'),
            "Total Trades": len(trades_df),
            "Total Closed Trades": len(trades_df),
            "Total Open Trades": 0}
    
    # Calculate win rate
    winning_trades = trades_df[trades_df['pnl_(USD)'] > 0]
    losing_trades = trades_df[trades_df['pnl_(USD)'] < 0]
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    
    # Best and worst trades
    best_trade = trades_df['return_(%)'].max() if not trades_df.empty else float('nan')
    worst_trade = trades_df['return_(%)'].min() if not trades_df.empty else float('nan')
    
    # Average winning and losing trades
    avg_winning_trade = winning_trades['return_(%)'].mean() if not winning_trades.empty else float('nan')
    avg_losing_trade = losing_trades['return_(%)'].mean() if not losing_trades.empty else float('nan')
    
    # Average durations
    avg_winning_duration = winning_trades['duration_(days)'].mean() if not winning_trades.empty else pd.NaT
    avg_losing_duration = losing_trades['duration_(days)'].mean() if not losing_trades.empty else pd.NaT
    
    # Profit factor (gross profits / gross losses)
    gross_profits = winning_trades['pnl_(USD)'].sum() if not winning_trades.empty else 0
    gross_losses = abs(losing_trades['pnl_(USD)'].sum()) if not losing_trades.empty else 0
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('nan')
    
    # Expectancy: (Win% * Avg Win) - (Loss% * Avg Loss)
    win_percent = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
    loss_percent = len(losing_trades) / len(trades_df) if len(trades_df) > 0 else 0
    
    avg_win_amount = winning_trades['pnl_(USD)'].mean() if not winning_trades.empty else 0
    avg_loss_amount = abs(losing_trades['pnl_(USD)'].mean()) if not losing_trades.empty else 0
    
    expectancy = (win_percent * avg_win_amount) - (loss_percent * avg_loss_amount)
    
    return {
        'Win Rate [%]': win_rate,
        'Best Trade [%]': best_trade,
        'Worst Trade [%]': worst_trade,
        'Avg Winning Trade [%]': avg_winning_trade,
        'Avg Losing Trade [%]': avg_losing_trade,
        'Avg Winning Trade Duration': avg_winning_duration,
        'Avg Losing Trade Duration': avg_losing_duration,
        'Profit Factor': profit_factor,
        'Expectancy': expectancy,
        "Total Trades": len(trades_df),
        "Total Closed Trades": len(trades_df),
        "Total Open Trades": 0}

def run_backtest_with_tax(
    price: pd.Series,
    orders: pd.Series,
    aud: pd.Series,
    init_cash: float,
    income: Union[float, list],
    trading_fees: float = 0.0,
    deductions: Union[float, list] = 0,
    tax_rates: pd.DataFrame = None,
    freq: str = '1D',
    special_conditions: dict = {"constant_cgt_discount": None,
                                "fixed_cgt_rate": None,
                                },
) -> vbt.Portfolio:
    """
    Run a backtest with tax calculations.
    
    Parameters
    ----------
    price : pd.Series
        Price series for the asset being traded
    orders : pd.Series
        Orders series (1 for buy, -1 for sell, 0 for hold)
    aud : pd.Series
        AUD exchange rate series
    init_cash : float
        Initial cash amount
    income : Union[float, list]
        Annual income for tax calculations. If float, same income for all years.
        If list, income for each year in order.
    deductions : Union[float, list], optional
        Annual deductions for tax calculations. If float, same deductions for all years.
        If list, deductions for each year in order. Default is 0.
    tax_rates : pd.DataFrame, optional
        Tax rates to use. If None, uses default Australian tax rates.
    freq : str, optional
        Frequency of the data. Default is '1D'.
    exit_at_end : bool, optional
        If True, exit all positions at the end of the backtest. Default is False.
    pay_final_tax_at_end : bool, optional
        If True, pay the tax for the final year at the end of the backtest. Default is False.

    special_conditions : dict, optional
        Dictionary of special conditions for the backtest. Keyword arguments, can be:
        - "

    Returns
    -------
    tuple
        (non_taxed_portfolio, taxed_portfolio, details_dict)
        - non_taxed_portfolio: vbt.Portfolio without tax adjustments
        - taxed_portfolio: vbt.Portfolio with tax adjustments
        - details_dict: Dictionary containing additional information
    """

    # Align date ranges
    common_dates = price.index.intersection(aud.index)
    if len(common_dates) < len(price.index):
        print(f"Warning: {len(price.index) - len(common_dates)} dates in price data not found in AUD data")
        print(f"Price data range: {price.index[0]} to {price.index[-1]}")
        print(f"AUD data range: {aud.index[0]} to {aud.index[-1]}")
    
    # Filter to common dates
    price = price[common_dates]
    orders = orders[common_dates]
    aud = aud[common_dates]
    
    # Get unique years in the data
    years = price.index.year.unique()
    #income and deductions are lists of length years
    if isinstance(income, list):
            income_series = pd.Series(income, index=years)
    else:
        income_series = pd.Series([income for _ in range(len(years))], index=years)
    if isinstance(deductions, list):
        deductions_series = pd.Series(deductions, index=years)
    else:
        deductions_series = pd.Series([deductions for _ in range(len(years))], index=years)

    # Initialize portfolio tracking dictionaries
    portfolios = {
        "pf": {
            "current_cash": init_cash,
            "current_assets": 0,
            "end_value": init_cash,
            "values_by_year": {},
            "order_at_start_of_year": np.nan,
            "all_trades": pd.DataFrame(),
            "sep_trades": {},
            "raw_trades": pd.DataFrame(),
            "all_orders": pd.DataFrame(),
            "portfolio_value": pd.Series(index=price.index, dtype=float)
        },
        "taxed_pf": {
            "current_cash": init_cash,
            "current_assets": 0,
            "end_value": init_cash,
            "values_by_year": {},
            "order_at_start_of_year": np.nan,
            "all_trades": pd.DataFrame(),
            "sep_trades": {},
            "raw_trades": pd.DataFrame(),
            "all_orders": pd.DataFrame(),
            "portfolio_value": pd.Series(index=price.index, dtype=float),
            "tax_details": pd.Series(),
            "tax_payments_(aud)": pd.Series(0, index=years, dtype=float),
            "tax_payments_(usd)": pd.Series(0, index=years, dtype=float),
            "capital_loss_carryforward": 0
        }
    }

    fys = []
    print(f"Backtest price data runs from: {price.index[0]} to {price.index[-1]}, years: {years}")

    for i, year in enumerate(years):
        if i == 0:  # First year
            if price.index[0].month < 7:
                # If starts before July, use from start to June 30th
                start_date = price.index[0]
                end_date = pd.Timestamp(f"{year}-06-30")
            else:
                # If starts after July, use from July 1st to June 30th next year
                start_date = pd.Timestamp(f"{year}-07-01")
                end_date = pd.Timestamp(f"{year+1}-06-30")    
        elif i == len(years)-1:  # Last year
            if price.index[-1].month > 6:
                # If ends after June, use from June 30th to end
                start_date = pd.Timestamp(f"{year}-06-30")
                end_date = price.index[-1]
            else:
                # If ends before July, use from June 30th previous year to end
                start_date = pd.Timestamp(f"{year-1}-06-30")
                end_date = price.index[-1]
        else:  # Middle years
            # Full financial year from June 30th to June 30th
            start_date = pd.Timestamp(f"{year-1}-06-30")
            end_date = pd.Timestamp(f"{year}-06-30")
            
        # Get the date range and append to fys
        year_mask = (price.index >= start_date) & (price.index <= end_date)
        fys.append(price.index[year_mask])
        print(f"For financial year: {year}\nStart date: {fys[i][0]}\nEnd date: {fys[i][-1]}")
    
    # Run backtest for each year
    deferred = pd.DataFrame()  # Placeholder for deferred tax
    for i, fy in enumerate(fys):
        # Get data for this year
        year_aud = aud[fy]
        year = fy[-1].year
        year_price = price[fy]
        year_orders = orders[fy]  # Fill NaN values with 0 (hold)

        print(f"\nRunning backtest for financial year: {year}")
        print(f"Order dates for {year}: {year_orders.dropna().index.tolist()}")
        
        # Process both portfolios
        for pf_type in ["pf", "taxed_pf"]:
            pf_data = portfolios[pf_type]
            
            year_deductions = deductions_series.loc[year]
            
            # Handle orders at start of year if needed
            current_orders = year_orders.copy()
            if not pd.isna(pf_data["order_at_start_of_year"]):
                current_orders = pd.concat([pd.Series(pf_data["order_at_start_of_year"], index=[year_price.index[0]]), 
                                           current_orders.iloc[1:]])
                print(f"Added order at start of year {year} for {pf_type}: {pf_data['order_at_start_of_year']}")
            
            # Run backtest for this year
            print(f"Before backtest, {pf_type} - Current cash: {pf_data['current_cash']}, current assets: {pf_data['current_assets']}, current value: {pf_data['end_value']}")
            print(f"Order values for {pf_type}: {current_orders.value_counts()}")
            
            pf = vbt.Portfolio.from_orders(
                year_price,
                size=current_orders,
                size_type='target_percent',
                init_cash=pf_data['end_value'],
                fees=trading_fees,
                freq=freq
            )
            
            # Store portfolio values for this year
            pf_data["values_by_year"][year] = pf.value()

            # Get end of year values
            end_cash = pf.cash().iloc[-1]
            end_assets = pf.assets().iloc[-1]
            end_price = year_price.iloc[-1]
            end_asset_fraction = (pf_data["current_assets"]*end_price) / pf_data["end_value"]
            end_cash_fraction = 1 - end_asset_fraction
            pf_data['current_cash'] = end_cash
            pf_data['current_assets'] = end_assets
            pf_data['end_value'] = end_cash + (end_assets * end_price)

            # Process orders
            ords = pd.DataFrame(pf.orders.records_readable)
            if not ords.empty:
                pf_data["all_orders"] = pd.concat([pf_data["all_orders"], ords], axis=0)
            
            # Get trades for this year
            print(f"New trades df for {year}, deferred trade left from last year: ")
            trades_df = create_trades_df(pf, year_price, year_aud, year_orders, deferred_trade=deferred)
            pf_data["raw_trades"] = pd.concat([pf_data["raw_trades"], trades_df], axis = 0)

            print(f"After backtest, {pf_type} - Current cash: {pf_data['current_cash']}, current assets: {pf_data['current_assets']}, current value: {pf_data['end_value']}")
            # If we have assets and there's a next year, prepare order for start of next year
            if i < len(fys) - 1 and pf_data["current_assets"] > 0.000000001:

                #Calculate target percentage for next year
                next_year_start = fys[i+1][0]
                # Calculate portfolio value at start of next year
                next_year_start_price = price.loc[next_year_start]
                portfolio_value_at_start = pf_data["current_cash"] + (pf_data["current_assets"] * next_year_start_price)
                
                # Calculate what percentage of portfolio value would give us the same number of assets
                target_percent = (pf_data["current_assets"] * next_year_start_price) / portfolio_value_at_start
                print(f"{pf_type} - Target percent for day 1 of year {year+1} order: {target_percent}")
                
                # Store the target percentage order for the first day of next year
                pf_data["order_at_start_of_year"] = target_percent
            else:
                # Reset order_at_start_of_year if we have no assets
                pf_data["order_at_start_of_year"] = np.nan
                print(f"{pf_type} - No assets at end of year {year}, no order will be placed at start of next FY")
            
            # For the taxed portfolio, handle tax calculations
            if pf_type == "taxed_pf":
                # Get AUD rate at year end for tax calculation
                year_end_aud = year_aud.iloc[-1]
                
                # Calculate tax
                year_income = income_series.loc[year]
                
                tax_result, deferred, trades_df = single_year_tax(year,
                    year_income,
                    trades_df,
                    year_end_aud,
                    previous_cg_loss=pf_data["capital_loss_carryforward"],
                    deductions=year_deductions,
                    tax_rates=tax_rates,
                    constant_cgt_discount=special_conditions["constant_cgt_discount"],
                    fixed_cgt_rate=special_conditions["fixed_cgt_rate"]
                )

                # Store tax payment
                pf_data["tax_details"] = pd.concat([pf_data["tax_details"], tax_result.rename(year)], axis=1)
                pf_data["tax_payments_(aud)"].loc[year] = tax_result['cgt_tax_(aud)']
                cgt_tax_usd = tax_result['cgt_tax_(aud)'] / year_aud.iloc[-1]
                pf_data["tax_payments_(usd)"].loc[year] = cgt_tax_usd
                
                # Update capital loss carryforward for next year from tax_result
                next_year_capital_loss = tax_result['capital_loss_carryforward_(AUD)']
                pf_data["capital_loss_carryforward"] = next_year_capital_loss
                if next_year_capital_loss > 0:
                    print(f"Capital loss of {next_year_capital_loss} AUD will be carried forward to next year")
            else:
                # For non-taxed portfolio, no tax adjustments needed
                cgt_tax_usd = 0
            
            # Handle year-end position and tax payment (for taxed portfolio)
            if pf_type == "taxed_pf" and cgt_tax_usd > 0:
                if end_cash >= cgt_tax_usd:
                    # If we have enough cash, just deduct the tax
                    pf_data["current_cash"] = end_cash - cgt_tax_usd
                    pf_data["current_assets"] = end_assets
                    pf_data["end_value"] = pf_data["current_cash"] + (pf_data["current_assets"] * end_price)
                    print(f"Paid CGT tax of {cgt_tax_usd} USD from available cash")
                else:
                    # If we don't have enough cash, sell assets to pay taxfolio
                    assets_to_sell = (cgt_tax_usd - end_cash) / end_price
                    #Calculate capital gains tax on the assets to sell, need the cost basis from the opening of the current trade....
                    #Get the last order made which'll have the cost basis data
                    ordas = pf.orders.records_readable
                    cb = ordas["Price"].iloc[-1] 
                    pnl_usd = (end_price - cb) * assets_to_sell
                    pnl_aud = pnl_usd * year_end_aud

                    print(f"{assets_to_sell:.6f} units of assets were sold, yielding {assets_to_sell*end_price:.2f} USD, to cover the CGT payment of {tax_result['cgt_tax_(aud)']:.2f} AUD/{cgt_tax_usd:.2f} USD.\
                          \nThis resulted in a cap gain of {pnl_aud:.2f} AUD. This will be added as a negative deduction to next financial year.\n \
                            \nCapital loss carryforward therefore updated to {pf_data['capital_loss_carryforward'] - pnl_aud:.2f} AUD")
                    pf_data["capital_loss_carryforward"] -= pnl_aud
                    
                    print(f"Paid CGT tax of {cgt_tax_usd} USD by selling {assets_to_sell} assets")

                    pf_data["current_assets"] = end_assets - assets_to_sell
                    pf_data["current_cash"] = 0.01  # Magic 1 cent tax refund
                    pf_data["end_value"] = pf_data["current_cash"] + (pf_data["current_assets"] * end_price)
    
            else:
                # For non-taxed portfolio or no tax due
                pf_data["current_cash"] = end_cash
                pf_data["current_assets"] = end_assets
                pf_data["end_value"] = end_cash + (end_assets * end_price)
 
            pf_data["sep_trades"][year] = trades_df
            if pf_data["all_trades"].empty:
                pass
            else:
                # Check if there are any trades and if the last trade has pay_cgt = "defer"
                if not pf_data["all_trades"].empty and pf_data["all_trades"].iloc[-1]["pay_cgt"] == "defer":
                    print("Dropping deferred cgt payment trade from the records and incorporating into next years tax instead.")
                    pf_data["all_trades"] = pf_data["all_trades"].iloc[:-1]
            pf_data["all_trades"] = pd.concat([pf_data["all_trades"], trades_df], axis=0)
    
    # Combine all year's portfolio values into one series for each portfolio
    for pf_type in ["pf", "taxed_pf"]:
        pf_data = portfolios[pf_type]
        for i, year in enumerate(years):
            year_mask = (price.index >= fys[i][0]) & (price.index <= fys[i][-1])
            if year in pf_data["values_by_year"]:
                pf_data["portfolio_value"].loc[year_mask] = pf_data["values_by_year"][year]
        
        # Adjust final portfolio value for the last year's tax payment if needed
        if pf_type == "taxed_pf" and pf_data["end_value"] != pf_data["portfolio_value"].iloc[-1]:
            # Get the last date in the portfolio value series
            last_date = pf_data["portfolio_value"].index[-1]
            # Update the final value to match the end_value which includes the tax payment
            pf_data["portfolio_value"].loc[last_date] = pf_data["end_value"]
            print(f"Adjusted final portfolio value for {pf_type} to {pf_data['end_value']} to account for final tax payment")
    
    # Create Portfolio objects from the value series
    non_taxed_portfolio = vbt.Portfolio.from_holding(
        portfolios["pf"]["portfolio_value"],
        init_cash=init_cash,
        freq=freq
    )
    
    taxed_portfolio = vbt.Portfolio.from_holding(
        portfolios["taxed_pf"]["portfolio_value"],
        init_cash=init_cash,
        freq=freq
    )

    # Calculate trade statistics for both portfolios
    non_taxed_trade_stats = calculate_trade_statistics(portfolios["pf"]["all_trades"])
    non_taxed_stats = non_taxed_portfolio.stats()
    for key in non_taxed_trade_stats.keys():
        non_taxed_stats[key] = non_taxed_trade_stats[key]
    taxed_trade_stats = calculate_trade_statistics(portfolios["taxed_pf"]["all_trades"])
    taxed_stats = taxed_portfolio.stats()
    for key in taxed_trade_stats.keys():
        taxed_stats[key] = taxed_trade_stats[key]
 
    if portfolios["taxed_pf"]["current_assets"] > 0.000001:
        taxed_stats["Total Open Trades"] = 1
        non_taxed_stats["Total Open Trades"] = 1
        non_taxed_stats["Total Closed Trades"]
        taxed_stats["Total Closed Trades"] = taxed_stats["Total Trades"] - 1

    # Prepare details dictionary
    details = {
        "non_taxed": {
            "portfolio_value": portfolios["pf"]["portfolio_value"],
            "all_trades": portfolios["pf"]["all_trades"],
            "sep_trades": portfolios["pf"]["sep_trades"],
            "raw_trades": portfolios["pf"]["raw_trades"],
            "all_orders": portfolios["pf"]["all_orders"],
            "current_cash": portfolios["pf"]["current_cash"],
            "current_assets": portfolios["pf"]["current_assets"],
            "end_value": portfolios["pf"]["end_value"],
            "stats": pd.Series(non_taxed_stats)
        },
        "taxed": {
            "portfolio_value": portfolios["taxed_pf"]["portfolio_value"],
            "all_trades": portfolios["taxed_pf"]["all_trades"],
            "sep_trades": portfolios["taxed_pf"]["sep_trades"],
            "raw_trades": portfolios["taxed_pf"]["raw_trades"],
            "all_orders": portfolios["taxed_pf"]["all_orders"],
            "current_cash": portfolios["taxed_pf"]["current_cash"],
            "current_assets": portfolios["taxed_pf"]["current_assets"],
            "end_value": portfolios["taxed_pf"]["end_value"],
            "tax_details": portfolios["taxed_pf"]["tax_details"].drop(0, axis = 1),
            "tax_payments_(aud)": portfolios["taxed_pf"]["tax_payments_(aud)"],
            "tax_payments_(usd)": portfolios["taxed_pf"]["tax_payments_(usd)"],
            "stats": pd.Series(taxed_stats)
        }
    }

    # Add trade statistics to details
    details["non_taxed"]["trade_stats"] = non_taxed_trade_stats
    details["taxed"]["trade_stats"] = taxed_trade_stats

    return non_taxed_portfolio, taxed_portfolio, details

def backtest_tax_cashflow(price: pd.Series, orders: pd.Series, init_cash: float, tax_payments_usd: pd.Series, freq: str = '1D') -> pd.DataFrame:
    """
    Run a vbt.Portolio.from_orders backtest of using the cgt_tax_pyments_(usd) series from the run_backtest_with_tax function.
    That function runs a backtest for each financial year with tax payments simulated. We can then use the cgt_tax_payments_(usd) series
    to run a final backtest using that series input to the cashflow parameter of the vbt.Portfolio.from_orders function. This will give us
    a final porftolio that has all the stats etc. caluclated and has tax payment simulation built in in an accurate way.

    **Parameters:**
    - price: pd.Series - Price series for the asset being traded
    - orders: pd.Series - Orders series (1 for buy with all cash, 0 for sell all assets, NaN for no change) - "size_type" = "target_percent" will be used.
    - init_cash: float - Initial cash amount
    - tax_payments_usd: pd.Series - The series of tax payments from the run_backtest_with_tax function. 
    - freq: str, optional - Frequency of the data. Default is '1D'.
    
    **Returns:**
    - vbt.Portfolio - The final portfolio object
    """
    
    pass


def calculate_performance_stats(equity_curve, periods_per_year=252, target_return=0.0):
    """
    Calculate performance statistics from equity curve time series
    
    Parameters:
    equity_curve: pd.Series with datetime index representing equity values
    periods_per_year: 252 for daily, 12 for monthly, 1 for annual
    target_return: target return for Sortino ratio (annualized)
    
    Returns:
    dict: Dictionary containing all performance statistics
    """
    
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Basic calculations
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    
    # Returns statistics
    mean_return = returns.mean()
    std_return = returns.std()
    
    # Winning and losing trades
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    
    # Profit Factor
    gross_profit = winning_returns.sum() if len(winning_returns) > 0 else 0
    gross_loss = abs(losing_returns.sum()) if len(losing_returns) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
    
    # Expectancy (average return per trade)
    expectancy = mean_return
    
    # Sharpe Ratio (assuming risk-free rate = 0)
    sharpe_ratio = (mean_return / std_return) * np.sqrt(periods_per_year) if std_return != 0 else 0
    
    # Maximum Drawdown for Calmar Ratio
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())
    
    # Calmar Ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0
    
    # Sortino Ratio
    target_return_periodic = target_return / periods_per_year
    downside_returns = returns[returns < target_return_periodic]
    downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
    sortino_ratio = (mean_return - target_return_periodic) / downside_deviation * np.sqrt(periods_per_year) if downside_deviation != 0 else 0
    
    # Omega Ratio
    threshold = target_return / periods_per_year
    excess_returns = returns - threshold
    gains = excess_returns[excess_returns > 0].sum()
    losses = abs(excess_returns[excess_returns < 0].sum())
    omega_ratio = gains / losses if losses != 0 else np.inf
    
    return {
        "End Value": equity_curve.iloc[-1],
        "Total Return [%]": total_return * 100,  # Convert to percentage
        'Profit Factor': profit_factor,
        'Expectancy': expectancy * 100,  # Convert to percentage
        'Sharpe Ratio': sharpe_ratio,
        'Calmar Ratio': calmar_ratio,
        'Omega Ratio': omega_ratio,
        'Sortino Ratio': sortino_ratio
    }

def orders_from_signals(signals: pd.Series, max_size: float = 1.0, 
                        value_map: dict = {0: 0, 0.5: 0.5, 1: 1}) -> pd.Series:
    """
    Convert a signals series to orders series.
    
    **Parameters:**
    -----------
    - signals : pd.Series
        Series with 1 for buy, -1 for sell, 0 for hold
    - max_size : float, optional
        max_size of any order in target percentage of portfolio terms. Default is 1.0.
    - value_map : dict, optional
        Mapping of signal values to order sizes. The keys could be strings, e.g "green", "yellow", "red"
        or numbers that differ from the fraction to replace the value with. The function will look for the 
        keys in the series and replace them with the values. Default is {0: 0, 0.5: 0.5, 1: 1}. 
    
        
    Returns:
    --------
    - orders: pd.Series
        Orders series with the same index as signals input series & only non-nan values on orders that should occur for a 
        vbt.Portfolio.from_orders(..) method run.
    - portfolio_holdings: pd.Series
        Orders series with the same index as signals input series & shows the potfolio asset holding proportion at each date.
    """
    
    portfolio_holdings = signals.copy()
    portfolio_holdings = portfolio_holdings.map(value_map)  # Map values according to value_map

    #Orders series that will contain a non-nan value only when a change to the portfolio holdings is made
    orders = orders_from_holdings(portfolio_holdings, max_size=max_size)

    return orders, portfolio_holdings

def orders_from_holdings(portfolio_holdings: pd.Series, max_size: float = 1.0) -> pd.Series:
    """
    Convert a portfolio holdings series to orders series.
    Takes a portfolio holdings series and converts it to an orders series that can be used in vbt.Portfolio.from_orders(..) method.
    Turns all values to NaN except for the values when a change in holdings occurs.
    """

    #Orders series that will contain a non-nan value only when a change to the portfolio holdings is made
    orders = portfolio_holdings.copy() 

    #Eliminate all values other than a change value (replace with nan)
    for i in range(len(portfolio_holdings[1:])):
        if portfolio_holdings.iloc[i] == portfolio_holdings.iloc[i-1]:
            orders.iloc[i] = np.nan
        else:
            pass
    return orders

if __name__ == "__main__":
    income = [91000, 93000, 96000, 103000, 112000, 117000]
    cgt_list = [1.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    deductions = [3500, 4200, 2200, 5000, 6000, 7500]

    
    # # Example usage
    # pf_with_tax = run_backtest_with_tax(
    #     price=data['BTCUSD'],
    #     entries=entries1,
    #     exits=exits1,
    #     aud=aud_series,  # Your USD/AUD exchange rate series
    #     init_cash=10000,
    #     income=[91000, 93000, 96000, 103000, 112000, 117000],  # Your income for each year
    #     cgt_rate=[1.0, 0.5, 1.0, 1.0, 1.0, 1.0],  # CGT rates for each year
    #     deductions=[3500, 4200, 2200, 5000, 6000, 7500],  # Your deductions for each year
    #     tax_rates=tax_rates_df  # Your tax rates DataFrame
    # )
