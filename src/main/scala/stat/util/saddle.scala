package stat.util

import org.saddle._
import stat._

object saddle {

  def normalize[R: ST: ORD, C: ST: ORD](f: Frame[R, C, Double]) =
    f.mapVec(v => v / v.stdev)

  def normalizeWith[R: ST: ORD, C: ST: ORD](f: Frame[R, C, Double],
                                            s: Series[C, Double]) = {
    Frame(f.toColSeq.map {
      case (cx, series) =>
        (cx, series / s.get(cx).get)
    }: _*)
  }

  def prep[R: ST: ORD](f: Frame[R, String, String],
                       missingMode: MissingMode,
                       forceCategorical: Set[String] = Set[String](),
                       naString: Set[String] = Set("na", "n/a", ""),
                       normalize: Boolean = false) = {
    import org.saddle.io._
    val categorical = nonNumericColumns(f, naString, forceCategorical)

    val trainingNumeric = missing(
      onehot(
        missingCategorical(f,
                           missingMode,
                           categorical.map(_._1).toSet ++ forceCategorical,
                           naString),
        categorical.map(_._1).toSet ++ forceCategorical)
        .mapValues(CsvParser.parseDouble),
      missingMode)

    val normed =
      if (normalize)
        trainingNumeric.mapVec(v => v / v.stdev)
      else trainingNumeric
    (categorical, normed)

  }

  def nonNumericColumns[R: ST: ORD](
      f: Frame[R, String, String],
      naString: Set[String] = Set("na", "n/a", ""),
      force: Set[String]): Seq[(String, Seq[String])] =
    f.toColSeq.filter {
      case (cx, series) =>
        force.contains(cx) || series.toVec.exists(
          d =>
            if (naString.contains(d.toLowerCase.trim)) false
            else
              try ({
                d.toDouble;
                false
              })
              catch { case _: NumberFormatException => true })
    }.map(x => x._1 -> x._2.toVec.toSeq.distinct)

  def onehot[T: ORD](v: Vec[T]): Seq[(T, Vec[Double])] =
    v.toSeq.distinct.map { a =>
      a -> v.map(x => if (x == a) 1.0 else 0.0)
    }.sortBy(_._1)

  def onehot[R: ST: ORD](d: Frame[R, String, String],
                         columns: Set[String]): Frame[R, String, String] = {
    val mapped: Seq[(String, Vec[String])] = d.toColSeq.flatMap {
      case (cx, series) =>
        if (!columns.contains(cx)) List((cx, series.toVec))
        else {
          onehot(series.toVec).drop(1).map {
            case (pivot, vec) =>
              (cx + "_" + pivot, vec.map(_.toString))
          }
        }
    }
    Frame[R, String, String](values = mapped.map(_._2),
                             rowIx = d.rowIx,
                             colIx = Index(mapped.map(_._1): _*))
  }

  def missingCategorical[R: ST: ORD](
      data: Frame[R, String, String],
      missingMode: MissingMode,
      nonNumericColumns: Set[String],
      naString: Set[String]
  ): Frame[R, String, String] = {
    val mapped = data.toColSeq.map {
      case (cx, series) =>
        if (!nonNumericColumns.contains(cx)) (cx, series)
        else {
          val filled: Series[R, String] = missingMode match {
            case MeanImpute =>
              val rep: String =
                series.toVec.toSeq.groupBy(x => x).toSeq.maxBy(_._2.size)._1
              series.mapValues(s =>
                if (naString.contains(s.toLowerCase.trim)) rep else s)
            case DropSample =>
              series.filter(s => !naString.contains(s.toLowerCase.trim))
          }
          (cx, filled)
        }
    }

    Frame(mapped: _*)
  }

  /**
    * Creates a completely filled in data matrix
    */
  def missing[I](
      data: Frame[I, String, Double],
      missingMode: MissingMode
  )(implicit ev: org.saddle.ST[I],
    ord: Ordering[I]): Frame[I, String, Double] =
    missingMode match {
      case MeanImpute =>
        data.mapVec(column => meanImpute(column))
      case DropSample =>
        data.rfilter(row => !(row.hasNA || row.toVec.exists(_.isInfinite)))
    }

  def meanImpute(data: Vec[Double]): Vec[Double] = {
    val mean = data.mean
    data.fillNA(_ => mean)
  }

}
