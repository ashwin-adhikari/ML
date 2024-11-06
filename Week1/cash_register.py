import json
import time

menu_file = r"C:\Users\Ripple\Desktop\ML\Week1\menu.json"
orders_file = r"C:\Users\Ripple\Desktop\ML\Week1\food_orders.json"
tax_rate = 0.13


def load_menu():
    try:
        with open(menu_file, "r") as file:
            return json.load(file) or {}
    except (json.JSONDecodeError, FileNotFoundError):
        print("Menu file is empty or corrupted.")
        return {}
    return {}


def save_menu(menu):
    with open(menu_file, "w") as file:
        json.dump(menu, file)


def load_orders():
    try:
        with open(orders_file, "r") as file:
            return json.load(file) or {}
    except (json.JSONDecodeError, FileNotFoundError):
        print("Orders file is empty or corrupted.")
        return {}
    return {}


def save_orders(orders):
    with open(orders_file, "w") as file:
        json.dump(orders, file)


menu = load_menu()
orders = load_orders()
current_order = []


def display_menu():
    print("Menu:\n")
    for item, price in menu.items():
        print(f"{item}: ${price:.2f}")
    print("\n")


def create_order():
    global current_order
    current_order = []
    print("Creating new order.\n")
    display_menu()


def add_item_to_order(menu, item, quantity):
    if item in menu:
        item_price = menu[item]
        order_item = {"name": item, "price": item_price, "quantity": quantity}
        current_order.append(order_item)  # Append works as current_order is a list now
        print(f"{quantity} x {item} added to order.")
    else:
        print("Item not found in the menu.")


def view_order():
    if not current_order:
        print("No items in order.")
    else:
        print("\nCurrent order:")
        for item in current_order:
            print(f"{item['name']} \t {item['quantity']} \t ${item['price']:.2f}")


def calculate_total():
    total = 0
    sub_total = sum([item["price"] * item["quantity"] for item in current_order])
    tax = sub_total * tax_rate
    total = sub_total + tax
    return sub_total, tax, total


def finalize_order():
    if not current_order:
        print("No items in order.")
    else:
        view_order()
        sub_total, tax, total = calculate_total()
        print(f"Subtotal: ${sub_total:.2f}")
        print(f"Tax: ${tax:.2f}")
        print(f"Total: ${total:.2f}")

        print("Please review your bill. Proceeding to payment in 10 seconds...\n")
        time.sleep(10)
        orders.append({
            "items": current_order.copy(),  # Use copy to save the current order state
            "subtotal": sub_total,
            "tax": tax,
            "total": total
        })
        save_orders(orders)  # Persist orders to file
        print("Order finalized and saved.\n")

        


def main():
    while True:
        print("1. Display Menu")
        print("2. Create Order")
        print("3. Add Item to Order")
        print("4. View Order")
        print("5. Finalize Order")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            display_menu()

        elif choice == "2":
            create_order()

        elif choice == "3":
            item = input("Enter item name: ")
            quantity = int(input("Enter quantity: "))
            add_item_to_order(menu, item, quantity)

        elif choice == "4":
            view_order()

        elif choice == "5":
            finalize_order()

        elif choice == "6":
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()