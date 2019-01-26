import org.saddle._
import org.saddle.linalg._

package object stat {
  implicit class PimpedDouble(d: Double) {
    def roundTo(i: Int) =
      (d * math.pow(10d, i)).round.toDouble * math.pow(10d, -1 * i)
  }

  def pearson(v1: Vec[Double], v2: Vec[Double] ) = {

    def findNA(v: Array[Double]) = {
      val ab = scala.collection.mutable.ArrayBuffer[Int]()
      var i = 0
      val n = v.size
      while (i < n) {
        if (v(i).isNaN) {
          ab.append(i)
        }
        i += 1
      }
      ab.toArray
    }

    val nanidx = findNA(v1) ++ findNA(v2)
    val v1wonan = v1.without(nanidx)
    val v2wonan = v2.without(nanidx)
    val v1dem = v1wonan.demeaned
    val v2dem = v2wonan.demeaned
    val v1s = v1dem.stdev
    val v2s = v2dem.stdev
    val cov = v1dem vv v2dem * (1d / (v1dem.length - 1))
    cov / (v1s * v2s)
  }

}
