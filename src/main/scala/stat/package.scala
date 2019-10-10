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

    val nanidx = findNA(v1.toArray) ++ findNA(v2.toArray)
    val v1wonan = v1.without(nanidx)
    val v2wonan = v2.without(nanidx)
    v1wonan.pearson(v2wonan)
  }

}
