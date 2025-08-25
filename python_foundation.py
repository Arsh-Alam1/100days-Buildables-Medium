print("Task 1. Variables & Data Types")
num_int = 10
num_float = 3.14
text_str = "Hello, Python!"
num_list = [1, 2, 3, 4, 5]

print("Integer:", num_int, type(num_int))
print("Float:", num_float, type(num_float))
print("String:", text_str, type(text_str))
print("List:", num_list, type(num_list))

print("\nTask 2. Conditionals")
number = int(input("Enter a number: "))
if number > 0:
    print("The number is positive.")
elif number < 0:
    print("The number is negative.")
else:
    print("The number is zero.")

print("\nTask 3. Loops")
print("First 10 natural numbers:")
for i in range(1, 11):
    print(i, end=" ")
print()

num = int(input("Enter a number to find factorial: "))
factorial = 1
for i in range(1, num + 1):
    factorial *= i
print("Factorial of", num, "is", factorial)

print("\nTask 4. Functions")
def greet(name):
    print(f"Hello, {name}!")

def is_even(num):
    return num % 2 == 0

greet("Arsh")
print("Is 4 even?", is_even(4))
print("Is 7 even?", is_even(7))

print("\nTask 5. List Operations")
numbers = [10, 5, 8, 3, 12]

print("List:", numbers)
print("Max:", max(numbers))
print("Min:", min(numbers))
print("Sum:", sum(numbers))
print("Average:", sum(numbers)/len(numbers))

numbers.sort()
print("Sorted List:", numbers)

numbers.reverse()
print("Reversed List:", numbers)

print("\nTask 6. Input/Output")
name = input("Enter your name: ")
age = input("Enter your age: ")
print(f"Hello {name}, you are {age} years old.")

