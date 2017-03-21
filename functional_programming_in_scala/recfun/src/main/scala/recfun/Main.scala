package recfun

object Main {
  def main(args: Array[String]) {
    println("Pascal's Triangle")
    for (row <- 0 to 10) {
      for (col <- 0 to row)
        print(pascal(col, row) + " ")
      println()
    }
  }

  /**
    * Exercise 1
    */
  def pascal(c: Int, r: Int): Int = {
    assert (c <= r)
    if (r == 0) 1
    else {
      if (c == 0) pascal(c, r - 1)
      else if (c == r) pascal(c - 1, r - 1)
      else pascal(c, r - 1) + pascal(c - 1, r - 1)
    }
  }

  /**
    * Exercise 2
    */
  def balance(chars: List[Char]): Boolean = {

    def go_to_next_par(sub_string: List[Char]): List[Char] = {
      if (sub_string.isEmpty || sub_string.head == '(' || sub_string.head == ')')
        sub_string
        else go_to_next_par(sub_string.tail)
    }

    def am_i_balanced(sub_string: List[Char], opened_count: Int): Boolean = {
      val current_string = go_to_next_par(sub_string)
      if (current_string.isEmpty) {
        if (opened_count == 0) true
        else false
      }
      else {
        val next_parenthesis = current_string.head
        if (next_parenthesis == ')')
          if (opened_count > 0)
            am_i_balanced(current_string.tail, opened_count - 1)
          else false
        else am_i_balanced(current_string.tail, opened_count + 1)
      }
    }

    am_i_balanced(chars, 0)
  }

  /**
    * Exercise 3
    */
  def countChange(money: Int, coins: List[Int]): Int = {
    def countCombinations(money: Int, coins: List[Int]): Int = {
      if (coins.isEmpty || money < 0) 0
      else if (money == 0 || (coins.length == 1 && money % coins(0) == 0)) 1
      else {
        countCombinations(money - coins.head, coins) + countChange(money, coins.tail)
      }
    }
    if (money == 0) 0
    else countCombinations(money, coins)
  }
}
