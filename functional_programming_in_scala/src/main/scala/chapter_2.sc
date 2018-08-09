// had to look it up :-(

object ex_1 {

  def fibonacci(n: Int): Int = {
    @annotation.tailrec
    def go(n: Int, prev: Int, curr: Int): Int =
      if (n == 0) prev
      else go(n - 1, curr, prev + curr)
    go(n, 0, 1)
  }
  // example
  (1 until 10) map fibonacci
}

