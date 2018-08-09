// had to look it up :-(

object chapter_2 {

  def fibonacci(n: Int): Int = {
    @annotation.tailrec
    def go(n: Int, prev: Int, curr: Int): Int =
      if (n == 0) prev
      else go(n - 1, curr, prev + curr)
    go(n, 0, 1)
  }
  // example
  (1 until 10) map fibonacci


  def isSorted[A](as: Array[A], ordered: (A, A) => Boolean): Boolean ={
    @annotation.tailrec
    def go(n: Int): Boolean = {
      if (n >= as.length)
        true
      else
        if (ordered(as(n), as(n - 1)))
          go(n + 1)
        else
          false

    }
    go(1)
  }

  val a = Array(1, 2, 3, 10)
  val b = Array(4, 2, 1, 10)

  isSorted(a, (x1: Int, x2: Int) => x1 >= x2)
  isSorted(b, (x1: Int, x2: Int) => x1 >= x2)
  
}