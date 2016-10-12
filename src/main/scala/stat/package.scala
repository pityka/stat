package object stat {
  implicit class PimpedDouble(d: Double) {
    def roundTo(i: Int) =
      (d * math.pow(10d, i)).round.toDouble * math.pow(10d, -1 * i)
  }
}
