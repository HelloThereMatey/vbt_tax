from typing import Union
import os
import sys
import pandas as pd
import vectorbt as vbt
import numpy as np
import plotly.graph_objects as go

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

def plot_portfolios(portfolios: dict):
    """Compare the value curves of multiple portfolios.

    **Parameters:**
    - portfolios: dict - A dictionary of portfolio objects to compare. The keys are the names of the portfolios and the values are the portfolio objects.
    """
    # Create figure
    fig = go.Figure()

    for name, pf in portfolios.items():
        fig.add_trace(
            go.Scatter(
                x=pf.value().index,
                y=pf.value().values,
                name=name,
                mode='lines',
                line=dict(width=1.5)
            )
        )

    # Update layout with horizontal legend at bottom
    fig.update_layout(
        title='BTC Trading using 42 MACRO VAMS Signal',
        yaxis_title='Portfolio Value ($)',
        template='plotly_white',
        width=1200,
        height=600,
        yaxis_type='log',
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    # Show the figure
    fig.show()


#FUNCTIONS #########################################################################################################################
def defaultTaxRates() -> pd.DataFrame:
    print("Using default tax information for Australia from 2019 - 2025.")
    taxRates = pd.read_excel(wd+fdel+"tax_rates.xlsx", index_col=0)
    # print(taxRates)
    return taxRates

def create_trades_df(pf: vbt.Portfolio, price: pd.Series, aud: pd.Series) -> pd.DataFrame:
    """ Add more info to the trades dataframe from a Portfolio backtest object.

    **Parameters:**
    - pf: vbt.Portfolio - The portfolio object to get the trades from.
    - price: pd.Series - The price series used for the backtest.
    - aud: pd.Series - The USD/AUD price history for conversion of pnl to AUD.

    **Returns:**
    - pd.DataFrame - The trades dataframe with more info.
    """
    # Check if we have any trades
    if len(pf.trades.records) == 0:
        return pd.DataFrame()  # Return empty dataframe if no trades

    trades = pf.trades.records.drop(columns=["col", "id", "direction", "status", "parent_id"])
    trades['entry_date'] = price.index[trades['entry_idx']]
    trades['exit_date'] = price.index[trades['exit_idx']]
    trades["return"] *= 100
    trades["duration_(days)"] = (trades['exit_date'] - trades['entry_date']).dt.days
    trades.rename(columns={"return": "return_(%)", "pnl": "pnl_(USD)"}, inplace=True)
    
    # Debug information
    print(f"Number of trades: {len(trades)}")
    print(f"Price date range: {price.index[0]} to {price.index[-1]}")
    print(f"AUD date range: {aud.index[0]} to {aud.index[-1]}")
    print(f"Trades date range: {trades['exit_date'].min()} to {trades['exit_date'].max()}")
    
    # Check if we have AUD data for all trade dates
    missing_dates = trades['exit_date'][~trades['exit_date'].isin(aud.index)]
    if len(missing_dates) > 0:
        print(f"Warning: Missing AUD rates for {len(missing_dates)} trade dates")
        print("First few missing dates:", missing_dates.head())
    
    # Using pandas asof to match dates (will use the most recent rate if exact match not found)
    trades["pnl_(AUD)"] = trades["pnl_(USD)"] * aud.reindex(trades['exit_date'], method='ffill').values
    
    trades["cgt_rate"] = 1.0
    for i in range(len(trades)):
        trades.loc[i, "financial_year"] = int(trades.loc[i, "exit_date"].year) if trades.loc[i, "exit_date"].month < 7 else int(trades.loc[i, "exit_date"].year) + 1
        if trades.loc[i, "duration_(days)"] > 365 and trades.loc[i, "pnl_(AUD)"] > 0:
            trades.loc[i, "cgt_rate"] = 0.5
        else:
            trades.loc[i, "cgt_rate"] = 1.0
    trades["cgt_amount_(AUD)"] = trades["pnl_(AUD)"] * trades["cgt_rate"]
    trades['financial_year'] = trades['financial_year'].astype(int)
    return trades

def tax_calc(trades: pd.DataFrame,
             income: Union[float, list], 
             deductions: Union[float, list] = 0, 
             tax_rates: pd.DataFrame = None) -> pd.DataFrame:
    """
    Calculate the tax payable for each financial year based on the trades and tax rates and income etc.

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

    ## The basic bits
    fy_totals = pd.Series(trades.groupby('financial_year')["cgt_amount_(AUD)"].sum().round(2)).to_frame()

    if tax_rates is None:
        tax_rates = defaultTaxRates()

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
    return fy_totals

def single_year_tax(year: int, income: float, year_trades: pd.DataFrame, usdaud: float, deductions: float = 0, tax_rates: pd.DataFrame = None) -> pd.Series:
    """Calculate the tax to pay for a single financial year from CGT and non-CGT income.
    
    **Parameters:**
    - year: int - The financial year (e.g. 2021 for 2020-21) - the year is the year the tax is paid.
    - income: float - Non-CGT income in AUD
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
        - total_tax_aud: Total tax payable
        - taxed_(% of gross income): Tax as percentage of gross income
        - non_cgt_tax_(aud): Tax on non-CGT income
        - cgt_tax_(aud): Tax on capital gains
    """

    if tax_rates is None:
        tax_rates = defaultTaxRates()

    # Calculate CGT amount from trades
    if year_trades.empty:
        cgt_aud = 0
    else:
        cgt_aud = year_trades["cgt_amount_(AUD)"].sum()
    
    # Calculate total taxable income
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
    cgt_tax = total_tax - non_cgt_tax
    
    # Create result series
    result = pd.Series({
        "cgt_amount_(AUD)": cgt_aud,
        "gross_taxable_income_(AUD)": gross_taxable_income,
        "gross_deductions_(AUD)": deductions,
        "net_taxable_income_(AUD)": net_taxable_income,
        "total_tax_aud": total_tax,
        "taxed_(% of gross income)": (total_tax / gross_taxable_income) * 100,
        "non_cgt_tax_(aud)": non_cgt_tax,
        "cgt_tax_(aud)": cgt_tax
    })
    
    return result

def run_backtest_with_tax(
    price: pd.Series,
    orders: pd.Series,
    aud: pd.Series,
    init_cash: float,
    income: Union[float, list],
    deductions: Union[float, list] = 0,
    tax_rates: pd.DataFrame = None,
    freq: str = '1D'
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
        
    Returns
    -------
    vbt.Portfolio
        A Portfolio object representing the tax-adjusted performance
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

    # Initialize variables
    portfolio_value = pd.Series(index=price.index, dtype=float)
    portfolio_value_taxed = pd.Series(index=price.index, dtype=float)
    cgt_tax_payments = pd.Series(0, index=years, dtype=float)
    current_cash = init_cash
    current_assets = 0
    
    # Dictionary to store values by year for final assembly
    portfolio_values_by_year = {}
    portfolio_values_taxed_by_year = {}

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
    
    order_at_start_of_year = np.nan
    end_value = init_cash
    all_trades = pd.DataFrame()
    all_orders = pd.DataFrame()

    # Run backtest for each year
    for i, fy in enumerate(fys):
        # Get data for this year
        # For Australian financial year (July 1st to June 30th)
        # If month is before July, it belongs to previous financial year
        year_aud = aud[fy]
        year = fy[-1].year
        year_price = price[fy]
        year_orders = orders[fy]

        print(f"\n\nRunning backtest for financial year: {year}")
        if not pd.isna(order_at_start_of_year):
            year_orders = pd.concat([pd.Series(order_at_start_of_year, index=[year_price.index[0]]), year_orders.iloc[1:]])
            print(f"Added order at start of year {year}: {order_at_start_of_year}, year_orders: \n{year_orders.head()}")

        # Run backtest for this year
        print(f"Current cash: {current_cash}, current assets: {current_assets}, current value: {end_value}")
        pf = vbt.Portfolio.from_orders(
            year_price,
            size = year_orders,
            size_type = 'target_percent',
            init_cash=end_value,
            freq=freq
        )

        print(f"Backtest for year {year} completed. Portfolio value: {pd.concat([pf.value(), year_price], axis=1)}")

        # Get trades for this year
        trades_df = create_trades_df(pf, year_price, year_aud)
        print(f"Trades for year {year}: {trades_df}")
        
        # Calculate tax for this year
        year_income = income_series.loc[year]
        year_deductions = deductions_series.loc[year]
            
        # Get AUD rate at year end for tax calculation
        year_end_aud = year_aud.iloc[-1]
        
        # Calculate tax
        tax_result = single_year_tax(
            year=year,
            income=year_income,
            year_trades=trades_df,
            usdaud=year_end_aud,
            deductions=year_deductions,
            tax_rates=tax_rates
        )
        
        print(f"Tax result for year {year}: {tax_result}")
        # Update portfolio value and tax payments
        portfolio_values_by_year[year] = pf.value()
        cgt_tax_payments.loc[year] = tax_result['cgt_tax_(aud)']  # Add tax at year end
        print(f"Paid total tax for {year} of {tax_result['total_tax_aud']} of which {cgt_tax_payments[year]} was CGT tax")
        
        # Store tax-adjusted values for this year
        tax_adjusted_values = pf.value().copy()
        # Adjust the final value to account for tax payment
        tax_adjusted_values.iloc[-1] -= tax_result['cgt_tax_(aud)']
        portfolio_values_taxed_by_year[year] = tax_adjusted_values
        
        # Get end of year values
        end_cash = pf.cash()[-1]
        end_assets = pf.assets()[-1]
        end_price = year_price[-1]
        cgt_tax = tax_result['cgt_tax_(aud)']
        
        # Handle year-end position and tax payment
        if end_cash >= cgt_tax:
            # If we have enough cash, just deduct the tax
            current_cash = end_cash - cgt_tax
            current_assets = end_assets
            end_value = current_cash + (current_assets * end_price)
        else:
            # If we don't have enough cash, sell assets to pay tax
            assets_to_sell = (cgt_tax - end_cash) / end_price
            current_assets = end_assets - assets_to_sell
            current_cash = 0.01  # Magic 1 cent tax refund
            end_value = current_cash + (current_assets * end_price)
            
        print(f"Current cash at end of year loop: {current_cash}, current assets: {current_assets}")
        all_trades = pd.concat([all_trades, trades_df], axis = 0)
        all_orders = pd.concat([all_orders, pf.orders.records], axis = 0)

        # If we have assets and there's a next year, add a buy order at start of next year
        if i < len(fys) - 1 and current_assets > 0:
            next_year_start = fys[i+1][0]
            # Calculate portfolio value at start of next year
            next_year_start_price = price.loc[next_year_start]
            portfolio_value_at_start = current_cash + (current_assets * next_year_start_price)
            
            # Calculate what percentage of portfolio value would give us the same number of assets
            target_percent = (current_assets * next_year_start_price) / portfolio_value_at_start
            
            # Create a target percentage order for the first day of next year
            order_at_start_of_year = target_percent
         
    # Combine all year's portfolio values into one series
    for i, year in enumerate(years):
        year_mask = (price.index >= fys[i][0]) & (price.index <= fys[i][-1])
        if year in portfolio_values_by_year:
            portfolio_value.loc[year_mask] = portfolio_values_by_year[year]
            portfolio_value_taxed.loc[year_mask] = portfolio_values_taxed_by_year[year]
    
    # Create a simulated asset price series from the tax-adjusted portfolio value
    # We'll use this to create a regular Portfolio
    simulated_price = portfolio_value.copy()
    simulated_price_taxed = portfolio_value_taxed.copy()
    
    # Create a regular Portfolio from the simulated price series
    # We'll use from_holding since we already have the value series
    pf_tax = vbt.Portfolio.from_holding(
        simulated_price,
        init_cash=init_cash,
        freq=freq
    )
    
    # Create a tax-adjusted portfolio
    pf_tax_adjusted = vbt.Portfolio.from_holding(
        simulated_price_taxed,
        init_cash=init_cash,
        freq=freq
    )
    
    details = {"portfolio_value": portfolio_value, 
               "portfolio_value_taxed": portfolio_value_taxed,
               "tax_payments": cgt_tax_payments, 
               "all_trades": all_trades, 
               "current_cash": current_cash, 
               "current_assets": current_assets,
               "all_orders": all_orders}
    return pf_tax, pf_tax_adjusted, details

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
