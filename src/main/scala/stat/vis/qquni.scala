package stat.vis

import org.nspl._
import org.nspl.data._
import org.nspl.awtrenderer._
import org.nspl.saddle._
import org.saddle._

object QQUniform {

  def writeQQPlot(fileName: String, pValues: Iterator[Double]): Unit =
    pngToFile(
      new java.io.File(fileName),
      plot(pValues).build,
      1000
    )

  def computeLambda(pvalues: Seq[Double],
                    minimum: Double = 0.0,
                    maximum: Double = 1.0): Double = {

    val medianObserved = pvalues.toVec.median

    (((maximum - minimum) * 0.5) + minimum) / medianObserved
  }
  def plot(pvalues: Iterator[Double],
           disablePruning: Boolean = false,
           minimum: Option[Double] = None,
           maximum: Option[Double] = None) = {

    if (pvalues.hasNext) {

      // The sole purpose of this scope is to let GC collect the array of doubles.
      val (lambda,
           data,
           first99percentLineEnd,
           last100LineEnd,
           expectedMax,
           last100) = {

        val observed = pvalues
          .filter(x => minimum.map(m => m < x).getOrElse(true))
          .toArray
          .sorted

        val N: Int = observed.size

        val majority = 0.99

        val first99percent: Int = ((1 - majority) * N).toInt

        val last1percent: Int = (majority * N).toInt

        val last100: Option[Int] = if (N > 100) Some(100) else None

        val total = (N / (maximum.getOrElse(1.0) - minimum.getOrElse(0.0)))

        val expectedMax = -1 * math.log10(1 / (total + 1).toDouble)

        val lambda = computeLambda(observed,
                                   minimum.getOrElse(0.0),
                                   maximum.getOrElse(1.0))

        val d99 = observed(first99percent)
        val d100 = last100.map(x => observed(x))

        // take random sample of 30k except the largest 10k p values
        val rand = new scala.util.Random(1234)
        val maxPlotted = 30000
        val randFilter = 1.0 / (N / maxPlotted.toDouble)

        val start = if (minimum.isEmpty) 0 else (minimum.get * N + 1).toInt

        val data: DataSource = (0 until N) flatMap { i =>
          val expected: Double = -1 * math.log10(
              (i + 1).toDouble / (total + 1) + minimum.getOrElse(0.0))
          val head = observed(i)
          if (disablePruning || i < 10000 || N < maxPlotted || rand.nextDouble < randFilter) {
            // mark for GC
            observed(i) = null.asInstanceOf[Double]

            // https://en.wikipedia.org/wiki/Order_statistic
            // the kth order statistic of the uniform distribution is a Beta random variable.[2][3]
            // ~ Beta(k,n+1-k).
            Some(
              (expected,
               -1 * math.log10(head),
               -1 * math.log10(
                 jdistlib.Beta.quantile(0.05, i, total + 1 - i, true, false)),
               -1 * math.log10(
                 jdistlib.Beta.quantile(0.95, i, total + 1 - i, true, false)))
            )
          } else None
        }

        (lambda, data, d99, d100, expectedMax, last100)
      }

      val first99percentLineData: DataSource = Seq(
        (0.0, -1 * math.log10(first99percentLineEnd)),
        (expectedMax, -1 * math.log10(first99percentLineEnd))
      )

      val last100LineData: Option[DataSource] = last100LineEnd.map {
        last100LineEnd =>
          List((0.0, -1 * math.log10(last100LineEnd)),
               (expectedMax, -1 * math.log10(last100LineEnd)))
      }

      val plot = xyplot(
        List(
          data -> List(
            line(xCol = 0, yCol = 2),
            line(xCol = 0, yCol = 3),
            point(xCol = 0,
                  yCol = 1,
                  size = 2f,
                  colorCol = 10,
                  sizeCol = 10,
                  shapeCol = 10)
          ),
          first99percentLineData -> List(line())
        ) ++ last100LineData.map(x => x -> List(line())): _*
      )(
        draw1Line = true,
        main = "lambda: " + lambda.toString,
        xlab = "Expected -log10 p-values",
        ylab = "Observed -log10 p-values",
        xgrid = false,
        ygrid = false
      )

      plot
    } else {
      xyplot(Nil)()

    }
  }

}
