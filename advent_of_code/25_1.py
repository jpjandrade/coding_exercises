def safe_password(rotations: list[str]):
    ans = 0
    pos = 50

    for rotation in rotations:
        direction = rotation[0]
        amount = int(rotation[1:])
        if direction == "L":
            amount *= -1
        pos = (pos + amount) % 100

        if pos == 0:
            ans += 1

    return ans


aoc_example = ["L68", "L30", "R48", "L5", "R60", "L55", "L1", "L99", "R14", "L82"]
assert safe_password(aoc_example) == 3

assert safe_password(["L2"]) == 0

assert safe_password(["R50", "L50", "L50"]) == 2

with open("25_1_input.txt", "r") as f:
    input_data = [line.strip() for line in f.readlines()]

print(safe_password(input_data))