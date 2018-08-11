

object chapter_2 {

  // ex 2.1 (had to look it up :-()

  def fibonacci(n: Int): Int = {
    @annotation.tailrec
    def go(n: Int, prev: Int, curr: Int): Int =
      if (n == 0) prev
      else go(n - 1, curr, prev + curr)
    go(n, 0, 1)
  }
  // example
  (1 until 10) map fibonacci

  // ex 2.2
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

  val arr1 = Array(1, 2, 3, 10)
  val arr2 = Array(4, 2, 1, 10)

  isSorted(arr1, (x1: Int, x2: Int) => x1 >= x2)
  isSorted(arr2, (x1: Int, x2: Int) => x1 >= x2)

  // exs 2.3, 2.4 and 2.5
  def curry[A, B, C](f: (A, B) => C): A => B => C = {
    a: A => (b: B) => f(a, b)
  }

  def uncurry[A, B, C](f: A => B => C): (A, B) => C = {
    (a: A, b: B) => f(a)(b)
  }

  // equivalent to f compose g
  def compose[A, B, C](f: B => C, g: A => B): A => C = {
    a: A => f(g(a))
  }



}