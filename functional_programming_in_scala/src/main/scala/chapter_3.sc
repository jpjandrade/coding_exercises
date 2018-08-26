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

  // ex 3.2
  def tail[A](xs: List[A]): List[A] = xs match {
    case Nil => Nil // or throw error?
    case Cons(_, x) => x
  }

  // ex 3.3
  def setHead[A](xs: List[A], a: A): List[A] = xs match {
    case Nil => Nil
    case Cons(_, x) => Cons(a, x) // or a :: xs.tail?
  }

  tail(l1)
  setHead(l2, 5.0)

  // ex 3.4
  def drop[A](l: List[A], n: Int): List[A] = {
    def go(xs: List[A], i: Int): List[A] =
      if (i == n)
        xs
    else
        go(tail(xs), i + 1)

    go(l, 0)
  }

  drop(l1, 2)

  // ex 3.5
  // had solved as "filter"" function then corrected it
  def dropWhile[A](l: List[A], f: A => Boolean): List[A] =
    l match {
      case Cons(h,t) if f(h) => dropWhile(t, f)
      case _ => l
    }

  def isOdd(n: Int): Boolean = {
    n % 2 == 1
  }

  dropWhile(l1, isOdd)

  def append[A](a1: List[A], a2: List[A]): List[A] =
    a1 match {
      case Nil => a2
      case Cons(h, t) => Cons(h, append(t, a2))
    }


  // ex 3.6
  // had to look it up :-(
  def init[A](l: List[A]): List[A] = {
    l match {
      case Cons(_, Nil) => Nil
      case Cons(x, h) => Cons(x, init(h))
      case Nil => Nil
    }
  }

  init(l1)

  def dropWhileCurried[A](l: List[A])(f: A => Boolean): List[A] =
    l match {
      case Cons(h,t) if f(h) => dropWhile(t, f)
      case _ => l
    }

  dropWhileCurried(l1)(_ % 2 == 1) // aahh much better

  def foldRight[A, B](as: List[A], z: B)(f: (A, B) => B): B = {
    as match {
      case Nil => z
      case Cons(x, xs) => f(x, foldRight(xs, z)(f))
    }
  }

  def sumFoldRight(ns: List[Int]): Int = {
    foldRight(ns, 0)((x, y) => x + y)
  }

  def productFoldRight(ns: List[Double]): Double = {
    foldRight(ns, 1.0)((x, y) => x * y)
  }

  // ex 3.7

  def productWithShortCircuit(ns: List[Double]): Double = {
    ns match {
      case Cons(0.0, _) => 0.0
      case _ => foldRight(ns, 1.0)(_ * _)
    }
  }

  productWithShortCircuit(l2)

  // ex 3.8
  foldRight(List(1, 2, 3), Nil: List[Int])(Cons(_, _))
  // this is just apply construction

  // ex 3.9
  def length[A](as: List[A]): Int = {
    foldRight(as, 0)((_, l) => l + 1)
  }

  length(l1)


  // ex 3.10
  def foldLeft[A, B](as: List[A], z: B)(f: (B, A) => B): B ={
    as match {
      case Nil => z
      case Cons(x, xs) => foldLeft(xs, f(z, x))(f)
    }
  }

  // ex 3.11
  def sumFoldLeft(ns: List[Int]): Int = {
    foldLeft(ns, 0)((x, y) => x + y)
  }

  def productFoldLeft(ns: List[Double]): Double = {
    foldLeft(ns, 1.0)((x, y) => x * y)
  }

  def lengthFoldLeft[A](as: List[A]): Int = {
    foldLeft(as, 0)((l, _) => l + 1)
  }

  sumFoldLeft(l1)

  productFoldLeft(l2)

  lengthFoldLeft(l2)

  // ex 3.12
  // looked it up because I was mixing List[A] with A
  def reverse[A](as: List[A]): List[A] = {
    foldLeft(as, List[A]())((xs, x) => Cons(x, xs))
  }

  reverse(l1)
  // ex 3.14
  def appendFold[A](a1: List[A], a2: List[A]): List[A] = {
    foldRight(a1, a2)((x, xs) => Cons(x, xs))
  }

  appendFold(l1, Cons(2, Cons(5, Nil)))
}