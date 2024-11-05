contacts = {}

def is_valid_input(name:str, number:int, email:str)->bool:
    if not name or not number or not email:
        print("Input cannot be empty.")
        return False
    if not isinstance(name, str):
        print("Name must be a string.")
        return False
    try:
        int(number)
    except ValueError:
        print("Number must be an integer.")
        return False
    if not isinstance(email, str) or "@" not in email or email.count("@") > 1:
        print("Invalid email format.")
        print("\nFormat: example@email")
        return False
    return True

def add(name:str, number:int, email:str):
    if is_valid_input(name, number, email):
        contacts[name] = {"number": int(number), "email": email}
        print("Contact added.")
    

def view():
    if not contacts:
        print("No contacts found.")
    for name, contact in contacts.items():
        print(f"Name: {name}\nNumber: {contact['number']}\nEmail: {contact['email']}")


def edit(name:str):
    if name in contacts:
        print(f"Editing contact '{name}'. Press Enter to keep existing values.")
        new_number = input(f"Enter new number (current: {contacts[name]['number']}): ")
        new_email = input(f"Enter new email (current: {contacts[name]['email']}): ")
        
        # Only update fields if a new value was provided
        if new_number:
            try:
                contacts[name]["number"] = int(new_number)
            except ValueError:
                print("Invalid number format. Update skipped for number.")
        if new_email:
            contacts[name]["email"] = new_email
        print("Contact updated.")
    else:
        print("Contact not found.")

def delete(name:str):
    if name in contacts:
        del contacts[name]
        print("Contact deleted.")
    else:
        print("Contact not found.")


def search(name:str):
    if name in contacts:
        contact = contacts[name]
        print(f"Name: {name}\nNumber: {contact['number']}\nEmail: {contact['email']}")
    else:
        print("Contact not found.")


if __name__ == "__main__":
    while True:
        print("1. Store a contact")
        print("2. Retrieve contacts")
        print("3. Edit a contact")
        print("4. Delete a contact")
        print("5. Search a contact")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            name = input("Enter name: ")
            number = input("Enter number: ")
            email = input("Enter email: ")
            add(name, number, email)

        elif choice == "2":
            view()

        elif choice == "3":
            name = input("Enter name: ")
            edit(name)

        elif choice == "4":
            name = input("Enter name: ")
            delete(name)

        elif choice == "5":
            search(input("Enter name: "))

        elif choice == "6":
            break

        else:
            print("Invalid choice. Please try again.")
