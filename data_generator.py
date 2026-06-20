import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

# Create Faker object
fake = Faker()

# Product catalog
PRODUCTS = [
    {"name": "Wireless Headphones", "category": "Electronics", "price": 89.99},
    {"name": "Running Shoes", "category": "Footwear", "price": 64.99},
    {"name": "Coffee Maker", "category": "Kitchen", "price": 49.99},
    {"name": "Yoga Mat", "category": "Sports", "price": 29.99},
    {"name": "Laptop Stand", "category": "Electronics", "price": 39.99},
    {"name": "Water Bottle", "category": "Sports", "price": 19.99},
    {"name": "Desk Lamp", "category": "Home", "price": 34.99},
    {"name": "Phone Case", "category": "Electronics", "price": 14.99},
]

STATUSES = ["completed", "pending", "shipped", "cancelled"]
STATUS_WEIGHTS = [0.65, 0.15, 0.15, 0.05]


def generate_orders(n_days=90, orders_per_day=20):
    """
    Generate n_days worth of order history.
    """
    orders = []
    start_date = datetime.now() - timedelta(days=n_days)

    for day in range(n_days):
        date = start_date + timedelta(days=day)

        # Weekends get 40% more orders
        daily_count = int(
            orders_per_day * (1.4 if date.weekday() >= 5 else 1.0)
        )

        # Add randomness (+/- 30%)
        daily_count = int(daily_count * random.uniform(0.7, 1.3))

        for _ in range(daily_count):
            product = random.choice(PRODUCTS)

            qty = random.choices(
                [1, 2, 3, 4],
                weights=[0.6, 0.25, 0.1, 0.05]
            )[0]

            status = random.choices(
                STATUSES,
                STATUS_WEIGHTS
            )[0]

            orders.append({
                "order_id": fake.uuid4()[:8].upper(),
                "date": date.strftime("%Y-%m-%d"),
                "customer": fake.name(),
                "email": fake.email(),
                "city": fake.city(),
                "country": fake.country(),
                "product": product["name"],
                "category": product["category"],
                "quantity": qty,
                "unit_price": product["price"],
                "revenue": round(product["price"] * qty, 2),
                "status": status,
            })

    return pd.DataFrame(orders)


def add_live_orders(df, n=3):
    """
    Simulate new orders arriving right now.
    """
    new_orders = []

    for _ in range(n):
        product = random.choice(PRODUCTS)
        qty = random.randint(1, 3)

        new_orders.append({
            "order_id": fake.uuid4()[:8].upper(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "customer": fake.name(),
            "email": fake.email(),
            "city": fake.city(),
            "country": fake.country(),
            "product": product["name"],
            "category": product["category"],
            "quantity": qty,
            "unit_price": product["price"],
            "revenue": round(product["price"] * qty, 2),
            "status": "pending",
        })

    return pd.concat(
        [df, pd.DataFrame(new_orders)],
        ignore_index=True
    )


# Test it
if __name__ == "__main__":
    df = generate_orders()

    print(f"Generated {len(df)} orders")
    print(f"Total revenue: ${df['revenue'].sum():,.2f}")
    print("\nSample Orders:")
    print(df.head(3))

    # Simulate live orders
    df = add_live_orders(df, n=3)

    print("\nAfter adding live orders:")
    print(f"Total orders: {len(df)}")
    print(df.tail(3))