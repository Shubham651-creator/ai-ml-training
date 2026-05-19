 
# ---- Step 1: Load the CSV file ----

orders = []  # empty list to store all orders

file = open("customer_orders.csv", "r")
lines = file.readlines()
file.close()

# Skip the first line (header row)
for line in lines[1:]:
    line = line.strip()           # remove newline character
    parts = line.split(",")       # split by comma

    order = {
        "order_id":      parts[0],
        "customer_name": parts[1],
        "product":       parts[2],
        "quantity":      int(parts[3]),
        "price":         float(parts[4]),
        "date":          parts[5]
    }
    orders.append(order)

print("Total orders loaded:", len(orders))


# ---- Step 2: Calculate revenue for each order ----

for order in orders:
    order["revenue"] = order["quantity"] * order["price"]


# ---- Step 3: Find insights ----

# 3a. Total revenue
total_revenue = 0
for order in orders:
    total_revenue += order["revenue"]

# 3b. Revenue per customer
customer_revenue = {}
for order in orders:
    name = order["customer_name"]
    if name in customer_revenue:
        customer_revenue[name] += order["revenue"]
    else:
        customer_revenue[name] = order["revenue"]

# 3c. Revenue per product
product_revenue = {}
for order in orders:
    product = order["product"]
    if product in product_revenue:
        product_revenue[product] += order["revenue"]
    else:
        product_revenue[product] = order["revenue"]

# 3d. Find top customer (highest revenue)
top_customer = ""
top_customer_revenue = 0
for name, revenue in customer_revenue.items():
    if revenue > top_customer_revenue:
        top_customer = name
        top_customer_revenue = revenue

# 3e. Find best-selling product (highest revenue)
best_product = ""
best_product_revenue = 0
for product, revenue in product_revenue.items():
    if revenue > best_product_revenue:
        best_product = product
        best_product_revenue = revenue


# ---- Step 4: Print the report ----

print()
print("========================================")
print("        CUSTOMER ORDERS REPORT          ")
print("========================================")
print("Total Orders     :", len(orders))
print("Total Revenue    : Rs.", total_revenue)
print()
print("--- Revenue by Customer ---")
for name, revenue in customer_revenue.items():
    print(" ", name, ":", "Rs.", revenue)
print()
print("--- Revenue by Product ---")
for product, revenue in product_revenue.items():
    print(" ", product, ":", "Rs.", revenue)
print()
print("--- Highlights ---")
print("  Top Customer   :", top_customer, "(Rs.", str(top_customer_revenue) + ")")
print("  Best Product   :", best_product, "(Rs.", str(best_product_revenue) + ")")
print("========================================")