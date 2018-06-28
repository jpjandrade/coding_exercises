class Rational(n: Int, d: Int){

  require(d != 0)

  private val g = gcd(n.abs, d.abs)

  val numer: Int = n / g
  val denom: Int = d / g

  def this(n: Int) = this(n, 1)



  override def toString: String = numer + "/" + denom

  private def gcd(a: Int, b: Int): Int =
    if (b == 0) a else gcd(b, a % b)

  def +(that: Rational): Rational = {
    new Rational(
      numer * that.denom + denom * that.numer,
      denom * that.denom)
  }

  def *(that: Rational) = {
    new Rational(numer * that.denom, denom * that.numer)
  }

  def lessThan(that: Rational) =
    this.numer * that.denom < that.numer * this.denom

  def max(that: Rational) =
    if(this lessThan that) that else this


}


val a = new Rational(2, 3)

val b =new Rational(5, 10)

a + b

val c = new Rational(66, 42)

c max (a + b)

c max a * b


