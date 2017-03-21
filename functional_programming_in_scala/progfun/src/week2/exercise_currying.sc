object exercise {
  def sum(f: Int => Int)(a: Int, b: Int): Int =
    if (a > b) 0
    else f(a) + sum(f)(a + 1, b)


  def product(f: Int => Int)(a: Int, b: Int): Int =
    if (a > b) 1
    else f(a) * product(f)(a + 1, b)
  println("Standard definitions")
  sum(x => x * x)(3, 4) // 3**2 + 4**2 = 25
  product(x => x * x)(3, 4) // 3**2 * 4**2 = 144

  def fact(n: Int): Int = {
    product(x => x)(1, n)
  }
  fact(5) // 5! = 120
  fact(4) // 4! = 24

  def generalFunction(agg: (Int, Int) => Int, neutralElement: Int)(f: Int => Int)(a: Int, b: Int): Int = {
    if (a > b) neutralElement
    else agg(f(a), generalFunction(agg, neutralElement)(f)(a + 1, b))
  }
  println("My general function")
  generalFunction((x, y) => x + y, 0)(x => x * x)(3, 4) // 25
  generalFunction((x, y) => x * y, 1)(x => x * x)(3, 4) // 144

  // actual solution from lecture

  def mapReduce(f: Int => Int, combine: (Int, Int) => Int, zero: Int)(a: Int, b: Int): Int =
    if (a > b) zero
    else combine(f(a), mapReduce(f, combine, zero)(a + 1, b))

  println("Course solution")
  mapReduce(x => x * x, (x, y) => x + y, 0)(3, 4)
  mapReduce(x => x * x, (x, y) => x * y, 1)(3, 4)
}