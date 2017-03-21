import math.abs

object exercise {
  val tolerance = 0.001

  def isCloseEnough(x: Double, y: Double) =
    abs((x - y) / x)  < tolerance

  def fixedPoint(f: Double => Double)(firstGuess: Double) = {
    def iterate(guess: Double): Double = {
      val next_guess = f(guess)
      if (isCloseEnough(guess, next_guess)) next_guess
      else iterate(next_guess)
    }
    iterate(firstGuess)
  }

  def averageDamp(f: Double => Double)(x: Double): Double =
    (x + f(x))/2

  def sqrtIter(x: Double): Double = {
    fixedPoint(averageDamp(y => x / y))(1.5)
  }
  sqrtIter(2)
}