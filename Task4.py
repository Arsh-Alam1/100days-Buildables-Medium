import math
from collections import Counter
def mean(numbers):
    return sum(numbers) / len(numbers) if numbers else 0
def median(numbers):
    n = len(numbers)
    if n == 0:
        return None
    numbers_sorted = sorted(numbers)
    mid = n // 2
    if n % 2 == 0:
        return (numbers_sorted[mid - 1] + numbers_sorted[mid]) / 2
    else:
        return numbers_sorted[mid]
def mode(numbers):
    if not numbers:
        return None
    freq = Counter(numbers)
    max_count = max(freq.values())
    modes = [k for k, v in freq.items() if v == max_count]
    return modes if len(modes) > 1 else modes[0]
def variance(numbers):
    if len(numbers) < 2:
        return 0
    m = mean(numbers)
    return sum((x - m) ** 2 for x in numbers) / len(numbers)
def std_dev(numbers):
    return math.sqrt(variance(numbers))
def euclidean_distance(x, y):
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y)
    elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))
    else:
        raise ValueError("Inputs must be numbers or lists of the same length")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))
if __name__ == "__main__":
    nums = [1, 2, 2, 3, 4,78,94,11,24,62]
    print("We will test the functions on [1,2,2,3,4,78,94,11,24,62]")
    print("Mean:", mean(nums))
    print("Median:", median(nums))
    print("Mode:", mode(nums))
    print("Variance:", variance(nums))
    print("Standard Deviation:", std_dev(nums))
    print("Euclidean Distance (values):", euclidean_distance(5, 2))
    print("Euclidean Distance (lists):", euclidean_distance([1,2,3],[4,5,6]))
    print("Sigmoid(0):", sigmoid(0))

"""
# What did I learn from these formulas?
Mean tells us the central tendency of data.
Median  shows the middle value, less affected by outliers.
Mode highlights the most frequent value(s).
Variance measures how spread out the data is.
Standard Deviation is the square root of variance, easier to interpret.
Euclidean Distance shows similarity/distance between numbers or vectors.
Sigmoid Function maps values into (0,1), useful in probability and machine learning.
"""

