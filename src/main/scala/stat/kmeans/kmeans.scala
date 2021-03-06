package stat.kmeans

import org.saddle._
import stat.sparse._

case class KMeansResult(clusters: Vec[Int],
                        means: IndexedSeq[Vec[Double]],
                        cost: Double)
