import json

inventory_file = r"C:\Users\Ripple\Desktop\ML\Week1\warehouse.json"
order_file = r"C:\Users\Ripple\Desktop\ML\Week1\orders.json"
low_stock_threshold = 5


def load_inventory():
    try:
        with open(inventory_file, "r") as file:
            return json.load(file) or {}
    except json.JSONDecodeError:
        print("Inventory file is empty or corrupted.")
        return {}
    return {}


def save_inventory(inventory):
    with open(inventory_file, "w") as file:
        json.dump(inventory, file)


def load_orders():
    try:
        with open(order_file, "r") as file:
            return json.load(file) or {}
    except json.JSONDecodeError:
        print("Orders file is empty or corrupted.")
        return []
    return []

def save_orders(orders):
    with open(order_file, "w") as file:
        json.dump(orders, file)


def add_item(inventory, id, name, quantity, price):
    if id in inventory:
        print("Item already exists.")
    else:
        inventory[id] = {"name": name, "quantity": quantity, "price": price}
        print("Item added.")
        save_inventory(inventory)


def update_item(inventory, id, name, quantity=None, price=None):
    if id in inventory:
        if quantity is not None:
            inventory[id]["quantity"] = quantity
        if price is not None:
            inventory[id]["price"] = price
        print("Item updated.")
        save_inventory(inventory)
    else:
        print("Item not found.")


def remove_item(inventory, id):
    if id in inventory:
        del inventory[id]
        print("Item removed.")
        save_inventory(inventory)
    else:
        print("Item not found.")


def view_inventory(inventory):
    if not inventory:
        print("Inventory is empty.")
    for id, details in inventory.items():
        low_stock_warning = (
            "(Low stock)" if details["quantity"] < low_stock_threshold else ""
        )
        print(
            f"ID: {id}\nName: {details['name']}\nQuantity: {details['quantity']} {low_stock_warning}\nPrice: {details['price']}\n"
        )


def add_order(inventory, orders, id, order_quantity):
    if id in inventory:
        if inventory[id]["quantity"] >= order_quantity:
            inventory[id]["quantity"] -= order_quantity
            orders.append({"id": id, "order_quantity": order_quantity})
            print(f"Order processed: {order_quantity} of {inventory[id]['name']}")
            save_inventory(inventory)
            save_orders(orders)
        else:
            print("Insufficient stock.")
    else:
        print("Item not found.")


def view_orders(inventory, orders):
    if not orders:
        print("No orders found.")
    for order in orders:
        print(
            f"ID: {order['id']}\nName: {inventory[order['id']]['name']}\nOrder Quantity: {order['order_quantity']}\n"
        )


def main():
    inventory = load_inventory()
    orders = load_orders()
    while True:
        print(
            "1. Add item\n2. Update item\n3. Remove item\n4. View inventory\n5. Add order\n6. View orders\n7. Exit"
        )
        choice = input("Enter choice: ")
        if choice == "1":
            id = input("Enter ID: ")
            name = input("Enter name: ")
            try:
                quantity = int(input("Enter quantity: "))
                price = float(input("Enter price: "))
                add_item(inventory, id, name, quantity, price)
            except ValueError:
                print("Invalid input.")

        elif choice == "2":
            id = input("Enter item ID to update: ")
            try:
                quantity = input("Enter new quantity: ")
                quantity = int(quantity) if quantity else None
                price = input("Enter new price: ")
                price = float(price) if price else None
                update_item(inventory, id, None, quantity, price)
            except ValueError:
                print("Invalid input.")

        elif choice == "3":
            id = input("Enter item ID to remove: ")
            remove_item(inventory, id)

        elif choice == "4":
            view_inventory(inventory)

        elif choice == "5":
            id = input("Enter ID: ")
            try:
                order_quantity = int(input("Enter order quantity: "))
                add_order(inventory, orders, id, order_quantity)
            except ValueError:
                print("Invalid input.")

        elif choice == "6":
            view_orders(inventory, orders)

        elif choice == "7":
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()