object rationals {
  val x = new Rational(1, 2)
  x.numer
  x.denom

  val y = new Rational(2, 3)
  x.add(y)
  y.toString()
}

class Rational(x: Int, y: Int) {
  def numer = x
  def denom = y

  def add(other: Rational) =
    new Rational(numer * other.denom + other.numer * denom,
      denom * other.denom)

  def toString(numer: Int, denom: Int): String = numer + "/" + denom
}

