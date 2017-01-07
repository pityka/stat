package stat.vis

import org.saddle._
import org.nspl._
import org.nspl.saddle._
import com.apporiented.algorithm.clustering._
import collection.JavaConversions._
import scala.reflect.ClassTag

object Heatmap {

  def fromColumns[RX <: AnyRef: ClassTag: ST: ORD,
                  CX <: AnyRef: ClassTag: ST: ORD](
      frame: Frame[RX, CX, Double],
      reorderRows: Boolean = true,
      euclidean: Boolean = false,
      main: String = "",
      xlab: String = "",
      ylab: String = "",
      xLabFontSize: Option[RelFontSize] = None,
      yLabFontSize: Option[RelFontSize] = None,
      mainFontSize: RelFontSize = 1 fts,
      colormap: Colormap = HeatMapColors(0, 1),
      zlim: Option[(Double, Double)] = None,
      valueText: Boolean = false,
      valueColor: Color = Color.black,
      valueFontSize: RelFontSize = 0.4 fts,
      tickLength: RelFontSize = 0.4 fts
  ) = {

    val distF = if (euclidean) euclideanDistance else angularDistance
    val (_, distmat, colindex) = clusterFrameByCols(frame)(distF)

    val maybe =
      if (reorderRows) Some(clusterFrameByCols(frame.T)(distF))
      else None

    val reorderedFrame =
      if (maybe.isDefined) frame.reindex(maybe.get._3, colindex)
      else frame.reindexCol(colindex)
    val reorderedDistmat = distmat.reindex(colindex, colindex)

    val heatmap = rasterplotFromFrame(reorderedFrame,
                                      main,
                                      xlab,
                                      ylab,
                                      xLabFontSize,
                                      yLabFontSize,
                                      mainFontSize,
                                      colormap,
                                      zlim = zlim,
                                      valueText = valueText,
                                      valueColor = valueColor,
                                      valueFontSize = valueFontSize,
                                      tickLength = tickLength)
    val distmap = rasterplotFromFrame(reorderedDistmat,
                                      main,
                                      xlab,
                                      xlab,
                                      xLabFontSize,
                                      xLabFontSize,
                                      mainFontSize,
                                      colormap,
                                      zlim = zlim,
                                      tickLength = tickLength)

    (heatmap, distmap, distmat)

  }

  def traverse[T](root: Cluster[T]): Seq[T] =
    if (root.isLeaf) Seq(root.getName)
    else root.getChildren.flatMap(x => traverse(x))

  def clusterFrameByCols[RX, CX <: AnyRef: ClassTag: ST: ORD](
      frame: Frame[RX, CX, Double])(distance: (Vec[Double],
                                               Vec[Double]) => Double) = {
    val names: Array[CX] = frame.colIx.toSeq.toArray
    val colseq = frame.toColSeq
    val distances = colseq.map {
      case (cxi, coli) =>
        colseq.map {
          case (cxj, colj) =>
            distance(coli.toVec, colj.toVec)
        }.toArray
    }.toArray
    val root = new DefaultClusteringAlgorithm().performClustering[CX](
      distances,
      names,
      new CompleteLinkageStrategy()
    )
    (root,
     Frame(Mat(distances), frame.colIx, frame.colIx),
     Index(traverse(root): _*))

  }

  val angularDistance = (s1: Vec[Double], s2: Vec[Double]) => {
    val inner = s1 dot s2
    val l1 = math.sqrt(s1 dot s1)
    val l2 = math.sqrt(s2 dot s2)
    val cosine = math.min(1d, inner / (l1 * l2))
    math.acos(cosine) / math.Pi
  }

  val euclideanDistance = (s1: Vec[Double], s2: Vec[Double]) => {
    var i = 0
    var sum = 0.0
    while (i < s1.length) {
      val diff: Double = s1.raw(i) - s2.raw(i)
      sum += diff * diff
      i += 1
    }
    math.sqrt(sum)
  }

}
