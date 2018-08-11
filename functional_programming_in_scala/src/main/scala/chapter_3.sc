import fpinscala.list.{Nil, Cons, List}
import fpinscala.list.List.{product, sum}

object chapter_3 {
  // sum example

  val l1 = List(1, 2, 3, 4)
  sum(l1)

  // prod example
  val l2 = List(1.0, 2.0, 3.0)
  product(l2)


  // ex 3.1

  val x = List(1, 2, 3, 4, 5) match {
    case Cons(x, Cons(2, Cons(4, _))) => x
    case Nil => 42
    case Cons(x, Cons(y, Cons(3, Cons(4, _)))) => x + y
    case Cons(h, t) => h + sum(t)
    case _ => 101
  }
  // my answer
  assert(x == 3)

  def tail[A](xs: List[A]): List[A] = xs match {
    case Nil => Nil // or throw error?
    case Cons(_, x) => x
  }

  def setHead[A](xs: List[A], a: A): List[A] = xs match {
    case Nil => Nil
    case Cons(_, x) => Cons(a, x) // or a :: xs.tail?
  }

  tail(l1)
  setHead(l2, 5.0)

  def drop[A](l: List[A], n: Int): List[A] = {
    def go(xs: List[A], i: Int): List[A] =
      if (i == n)
        xs
    else
        go(tail(xs), i + 1)

    go(l, 0)
  }

  drop(l1, 2)

}