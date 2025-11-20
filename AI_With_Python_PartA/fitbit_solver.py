import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_demand(price):
    """
    Calculates expected demand based on price.
    Demand = 208,000 - 502 * Price
    Demand cannot be negative.
    """
    demand = 208000 - 502 * price
    return max(0, demand)

def calculate_production_cost(quantity):
    """
    Calculates total production cost based on tiered marginal costs.
    
    Tiers:
    0 - 50,000: $75.00
    50,000 - 75,000: $55.00
    75,000 - 100,000: $40.00
    100,000 - 125,000: $28.00
    125,000+: $20.00
    """
    cost = 0
    remaining_q = quantity
    
    # Tier 1: First 50,000
    tier1_cap = 50000
    if remaining_q > 0:
        amount = min(remaining_q, tier1_cap)
        cost += amount * 75.00
        remaining_q -= amount
        
    # Tier 2: Next 25,000 (up to 75,000)
    tier2_cap = 25000
    if remaining_q > 0:
        amount = min(remaining_q, tier2_cap)
        cost += amount * 55.00
        remaining_q -= amount
        
    # Tier 3: Next 25,000 (up to 100,000)
    tier3_cap = 25000
    if remaining_q > 0:
        amount = min(remaining_q, tier3_cap)
        cost += amount * 40.00
        remaining_q -= amount
        
    # Tier 4: Next 25,000 (up to 125,000)
    tier4_cap = 25000
    if remaining_q > 0:
        amount = min(remaining_q, tier4_cap)
        cost += amount * 28.00
        remaining_q -= amount
        
    # Tier 5: Remainder
    if remaining_q > 0:
        cost += remaining_q * 20.00
        
    return cost

def calculate_production_time(quantity):
    """
    Determines production time based on the total quantity ordered.
    Time is determined by the last device produced.
    """
    if quantity <= 0:
        return 0
    elif quantity <= 50000:
        return 6
    elif quantity <= 75000:
        return 9
    elif quantity <= 100000:
        return 11
    elif quantity <= 125000:
        return 12
    else:
        return 13

def calculate_profit_details(price, announced_shipping_date):
    """
    Calculates profit and other details for a given price and announced shipping date.
    """
    demand = calculate_demand(price)
    revenue = price * demand
    
    prod_cost = calculate_production_cost(demand)
    actual_time = calculate_production_time(demand)
    
    delay = max(0, actual_time - announced_shipping_date)
    rebate_per_customer = 25 * delay
    total_rebate = demand * rebate_per_customer
    
    profit = revenue - prod_cost - total_rebate
    
    return {
        "Price": price,
        "Announced_Date": announced_shipping_date,
        "Demand": demand,
        "Revenue": revenue,
        "Production_Cost": prod_cost,
        "Actual_Time": actual_time,
        "Delay": delay,
        "Total_Rebate": total_rebate,
        "Profit": profit
    }

def main():
    # --- Part I: Specific Scenario ---
    print("--- Part I: Analysis for Price=$250, Announced Date=9 weeks ---")
    p1_price = 250
    p1_date = 9
    p1_results = calculate_profit_details(p1_price, p1_date)
    
    for key, value in p1_results.items():
        if "Cost" in key or "Revenue" in key or "Profit" in key or "Rebate" in key:
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value}")
    print("\n")

    # --- Part II: Optimize Price (assuming fixed date of 9 weeks) ---
    print("--- Part II: Optimizing Price (Fixed Announced Date = 9 weeks) ---")
    prices = np.arange(100, 401, 10) # $100 to $400 in increments of $10
    results_p2 = []
    
    for p in prices:
        res = calculate_profit_details(p, 9)
        results_p2.append(res)
        
    df_p2 = pd.DataFrame(results_p2)
    
    # Find max profit
    best_p2 = df_p2.loc[df_p2['Profit'].idxmax()]
    print(f"Optimal Price: ${best_p2['Price']}")
    print(f"Max Profit: ${best_p2['Profit']:,.2f}")
    
    # Plotting Part II
    plt.figure(figsize=(10, 6))
    plt.plot(df_p2['Price'], df_p2['Profit'], marker='o')
    plt.title('Profit vs Price (Announced Shipping: 9 Weeks)')
    plt.xlabel('Price ($)')
    plt.ylabel('Profit ($)')
    plt.grid(True)
    plt.axvline(x=best_p2['Price'], color='r', linestyle='--', label=f"Optimal Price: ${best_p2['Price']}")
    plt.legend()
    plt.savefig('part2_profit_vs_price.png')
    print("Saved graph: part2_profit_vs_price.png\n")

    # --- Part III: 2D Analysis (Price vs Announced Date) ---
    print("--- Part III: 2D Analysis (Price vs Announced Date) ---")
    shipping_dates = np.arange(4, 16, 1) # 4 to 15 weeks
    
    # We will store results to create a heatmap and find optimal strategy
    heatmap_data = []
    optimal_strategies = []

    for w in shipping_dates:
        best_profit_for_w = -float('inf')
        best_price_for_w = -1
        
        for p in prices:
            res = calculate_profit_details(p, w)
            heatmap_data.append({
                "Price": p,
                "Announced_Date": w,
                "Profit": res['Profit']
            })
            
            if res['Profit'] > best_profit_for_w:
                best_profit_for_w = res['Profit']
                best_price_for_w = p
        
        optimal_strategies.append({
            "Announced_Date": w,
            "Optimal_Price": best_price_for_w,
            "Max_Profit": best_profit_for_w
        })

    df_heatmap = pd.DataFrame(heatmap_data)
    pivot_table = df_heatmap.pivot(index='Price', columns='Announced_Date', values='Profit')
    
    # Plotting Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='viridis', annot=False, fmt=".0f")
    plt.title('Profit Heatmap: Price vs Announced Shipping Date')
    plt.gca().invert_yaxis() # Put lower prices at bottom if preferred, or keep standard matrix view. Standard is usually fine.
    plt.savefig('part3_profit_heatmap.png')
    print("Saved graph: part3_profit_heatmap.png")
    
    # Plotting Optimal Price vs Announced Date
    df_opt = pd.DataFrame(optimal_strategies)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_opt['Announced_Date'], df_opt['Optimal_Price'], marker='s', color='green')
    plt.title('Optimal Price for each Announced Shipping Date')
    plt.xlabel('Announced Shipping Date (Weeks)')
    plt.ylabel('Optimal Price ($)')
    plt.grid(True)
    plt.savefig('part3_optimal_price_vs_date.png')
    print("Saved graph: part3_optimal_price_vs_date.png")

    # Global Maximum
    global_best = df_heatmap.loc[df_heatmap['Profit'].idxmax()]
    print(f"\nGlobal Optimal Strategy:")
    print(f"Price: ${global_best['Price']}")
    print(f"Announced Date: {global_best['Announced_Date']} weeks")
    print(f"Profit: ${global_best['Profit']:,.2f}")

if __name__ == "__main__":
    main()
