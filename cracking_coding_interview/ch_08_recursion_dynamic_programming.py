import numpy as np

# ex 8.1


def stairs_comb(n):
    memo = []
    memo.append(0)  # appending is less explicit but takes care of corner cases (i.e., n = 1)
    memo.append(1)
    memo.append(2)
    memo.append(4)
    if n >= 4:
        for i in range(4, n + 1):
            combinations = memo[i - 1] + memo[i - 2] + memo[i - 3]
            memo.append(combinations)
    return memo[n]

# ex 8.2
# this one was pretty fun to solve
# edit: damn, totally missed the O(r ^{r + c}) -> O(rc) optimization :-(
# edited to include it but jesus that one sucked to miss


def generate_grid(r, c):
    G = (np.random.binomial(n=1, p=0.8, size=(r, c)) - 1).tolist()
    G[0][0] = 0
    G[-1][-1] = 1
    grid = {
        'G': G,
        'r': r,
        'c': c
    }
    return grid


def robot_on_a_grid(grid):
    initial_x = 0
    initial_y = 0
    path = []
    failed_points = set()
    final_path, found = robot_search(initial_x, initial_y, path, failed_points, grid)
    if found:
        return final_path
    else:
        return "No path found! :-("


def robot_search(x, y, path, failed_points, grid):
    G = grid['G']
    is_visited = (x, y) in failed_points
    if x >= grid['r'] or y >= grid['c'] or G[x][y] == -1 or is_visited:
        return path, False

    if G[x][y] == 1:
        return path, True

    failed_points_updated = failed_points.union({(x, y)})  # let's keep side effects to a minimal
    path_d, found_d = robot_search(x + 1, y, path + ['d'], failed_points_updated, grid)

    if found_d:
        return path_d, True

    path_r, found_r = robot_search(x, y + 1, path + ['r'], failed_points_updated, grid)

    return path_r, found_r


# ex 8.12

GRID_SIZE = 8

def place_queens(row, current_column_positions, results):
    if row == GRID_SIZE:
        results.append(current_column_positions[:])  # slice is for copying
    else:
        for col in range(GRID_SIZE):
            if check_valid(row, col, current_column_positions):
                current_column_positions.append(col)
                place_queens(row + 1, current_column_positions, results)
    return results

def check_valid(row, candidate_col, curr_cols):
    for past_row in range(row):
        past_col = curr_cols[past_row]
        if past_col == candidate_col:
            return False
        elif abs(past_row - row) == abs(past_col - candidate_col):
            return False

    return True

