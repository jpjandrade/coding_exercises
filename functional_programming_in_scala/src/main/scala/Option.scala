package fpinscala.option

sealed trait Option[+A]{
  def map[B](f: A => B): Option[B] = {
    case None => None
    case Some(d) => Some(f(d))
  }

  // had to look everything up besides map. I'm very confused right now, only knew how to do these with pattern matching :-P
  def flatMap[B](f: A => Option[B]): Option[B] = {
    map(f).getOrElse(None)
  }

  def getOrElse[B >: A](default: => B): B = {
    case None => default
    case Some(d) => d
  }

  // how the hell is this clearer than doing pattern matching?
  def orElse[B >: A](ob: => Option[B]): Option[B] = {
    this map (Some(_)) getOrElse ob
  }

  def filter(f: A => Boolean): Option[A] =
    flatMap(a => if(f(a)) Some(a) else None)
}

case class Some[+A](get: A) extends Option[A]
case object None extends Option[Nothing]