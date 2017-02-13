import org.saddle._
import org.saddle.linalg._
import stat._
import org.saddle.io._
import stat.util.saddle._
import stat.rbf._
import stat.crossvalidation._
import stat.sgd._

import org.nspl._
import org.nspl.saddle._
import org.nspl.data._
import org.nspl.awtrenderer._

import java.io.File

object RegressionExample extends App {

  val rng = new scala.util.Random(42)

  val removedRowIds =
    Set("524", "1299", "167", "310", "606", "643")

  val trainingF =
    CsvParser
      .parse(CsvFile(args(0)))
      .withColIndex(0)
      .withRowIndex(0)
      .rfilterIx(rx => !removedRowIds.contains(rx))

  val (categorical, trainingNumeric1) =
    prep(trainingF,
         MeanImpute,
         forceCategorical = Set("MSSubClass"),
         normalize = false)

  val categoricalVariables: Set[String] = categorical.map(_._1).toSet
  val continuousVariables = trainingF.colIx.toSeq
    .filterNot((c: String) => categoricalVariables.contains(c))
    .toSet
  // val (_, testNumeric) = prep(testF, MeanImpute)

  val trainingY =
    trainingNumeric1.firstCol("SalePrice").toVec.map(x => math.log10(x + 1))

  val removedColumns = Set("Utilities_NoSeWa",
                           "Condition2_RRAe",
                           "Condition2_RRAn",
                           "Condition2_RRNn",
                           "HouseStyle_2.5Fin",
                           "RoofMatl_CompShg",
                           "RoofMatl_Membran",
                           "RoofMatl_Metal",
                           "RoofMatl_Roll",
                           "Exterior1st_ImStucc",
                           "Exterior1st_Stone",
                           "Exterior2nd_Other",
                           "Heating_GasA",
                           "Heating_OthW",
                           "Electrical_Mix",
                           "GarageQual_Fa",
                           "PoolQC_Fa",
                           "PoolQC_Gd",
                           "MiscFeature_TenC")

  val trainingNumeric = Frame(
    trainingNumeric1.toColSeq
      .filterNot(_._1 == "SalePrice")
      .filterNot(x => removedColumns.contains(x._1)) :+ ("SalePrice" -> Series(
      trainingY,
      trainingNumeric1.rowIx)): _*).rfilterIx(rx =>
    !removedRowIds.contains(rx))

  val testF = CsvParser.parse(CsvFile(args(1))).withColIndex(0).withRowIndex(0)

  val (_, testNumeric) = prep(testF,
                              MeanImpute,
                              forceCategorical = Set("MSSubClass"),
                              normalize = false)

  println(trainingF.colIx.toSeq.toSet &~ testF.colIx.toSeq.toSet)

  println(trainingNumeric.colIx.toSeq.toSet &~ testNumeric.colIx.toSeq.toSet)

  val trainingX = trainingNumeric.filterIx(_ != "SalePrice")

  // pdfToFile(new File("hist_test_train.pdf"),
  //           sequence(trainingX.colIx.toSeq.map { cx =>
  //             val test = testNumeric.firstCol(cx).toVec
  //             val training = trainingNumeric.firstCol(cx).toV
  //
  //             xyplot(
  //               HistogramData(test.toSeq, 100) -> bar(
  //                 strokeColor = Color.green.copy(a = 100)),
  //               HistogramData(training.toSeq, 100) -> bar(
  //                 strokeColor = Color.red.copy(a = 100))
  //             )(xlab = cx)
  //           }, TableLayout(10)),
  //           1000)
  // val num = Frame(trainingNumeric.toColSeq.toList.filterNot(x =>
  //   categorical.exists(y => x._1.startsWith(y._1))): _*)
  //
  // pdfToFile(new File("qqnorm.pdf"), stat.vis.QQNorm(trainingY), 1000)
  // pdfToFile(new File("yhist.pdf"),
  //           xyplot(
  //             HistogramData(trainingY.toSeq, 100) -> bar()
  //           )(xlab = "y", ylab = "freq.", ylim = Some(0d -> Double.NaN)),
  //           1000)
  //
  // val boxplotsOfCategorical = sequence(categoricalVariables.toList.map { cx =>
  //   val d: Seq[(String, Double)] = (trainingF
  //     .firstCol(cx)
  //     .toVec
  //     .toSeq zip trainingF.firstCol("SalePrice").toVec.toSeq).map {
  //     case (c, sp) => (c, CsvParser.parseDouble(sp))
  //   }
  //   boxplotFromLabels(d, xLabelRotation = -0.5, main = cx)
  // }, TableLayout(10))
  // pdfToFile(new File("boxplots_cat.pdf"), boxplotsOfCategorical, 1000)
  //
  // val boxplotsOfCont = sequence(
  //   trainingX.toColSeq.toList
  //     .filter(x => continuousVariables.contains(x._1))
  //     .map {
  //       case (cx, series) =>
  //         val d: Seq[(Double, Double)] =
  //           (series.toVec.toSeq zip trainingY.toSeq).map {
  //             case (c, sp) => (c, sp)
  //           }
  //         boxplotFromLabels(d,
  //                           xLabelRotation = -0.5,
  //                           main = cx,
  //                           useLabels = false)
  //     },
  //   TableLayout(10))
  // pdfToFile(new File("boxplots_cont.pdf"), boxplotsOfCont, 1000)
  //
  // val correlationPlot = stat.vis.CorrelationPlot.fromColumns(trainingX)._1
  // pdfToFile(new File("corr.pdf"), correlationPlot, 1000)
  //
  // // val (heatmap, pairwise, _) =
  // //   stat.vis.Heatmap.fromColumns(trainingX, colormap = LogHeatMapColors())
  // // pdfToFile(new File("heatmap.pdf"),
  // //           group(heatmap, pairwise, TableLayout(2)),
  // //           1000)
  //
  // val histograms = {
  //   sequence(trainingX.toColSeq.toList
  //              .filterNot(x => categorical.exists(y => x._1.startsWith(y._1)))
  //              .map {
  //                case (cx, series) =>
  //                  xyplot(
  //                    HistogramData(series.toVec.toSeq, 100) -> bar()
  //                  )(xlab = cx, ylab = "freq.", ylim = Some(0d -> Double.NaN))
  //              },
  //            TableLayout(10))
  // }
  // pdfToFile(new java.io.File("hist.pdf"), histograms, 1000)
  //
  // // val trainingSamples = trainingNumeric.numRows
  //
  // val pca = stat.pca.fromData(trainingX, 5)
  // val pcaplot =
  //   stat.pca
  //     .plot(pca, 3, scala.util.Right(trainingNumeric.firstCol("SalePrice")))
  // pdfToFile(new java.io.File("pca.pdf"), pcaplot, 1000)
  //
  // println(pca.loadings.get.firstCol(0).sorted)
  // println(pca.loadings.get.firstCol(1).sorted)

  // println(trainingNumeric.toMat.contents.count(_.isNaN))
  // println(trainingNumeric.colIx.toSeq)

  slogging.LoggerConfig.factory = slogging.PrintLoggerFactory()
  slogging.LoggerConfig.level = slogging.LogLevel.DEBUG

  def extend(t: Frame[String, String, Double]) =
    Frame(t.toColSeq.flatMap {
      case (cx, series) =>
        if (!Set("GrLivArea",
                 "TotRmsAbvGrd",
                 "YearBuilt",
                 "Fireplaces",
                 //  "YearRemodAdd",
                 "GarageYrBlt").contains(cx))
          List((cx, series))
        else
          List(
            (cx, series),
            (cx + "_2", series.mapValues(x => x * x))
            // (cx + "_3", series.mapValues(x => x * x * x)),
            // (cx + "_4", series.mapValues(x => x * x * x * x))
          )
    }: _*)

  val marginallySignificantColumns = trainingNumeric
    .filterIx(cx => cx != "SalePrice")
    .toColSeq
    .map {
      case (cx, series) =>
        val lm1 = stat.regression.LinearRegression.linearRegression(
          data = trainingNumeric.col("SalePrice", cx),
          yKey = "SalePrice",
          lambda = 0,
          addIntercept = true
        )
        val p = lm1.covariates(cx)._2.pValue
        val b = lm1.covariates(cx)._1.slope
        if (p < 1E-4) {
          if (b > 0) Some((cx, 1))
          else Some((cx, -1))
        } else None
    }
    .filter(_.isDefined)
    .map(_.get)
    .toMap

  def sumCount(t: Frame[String, String, Double]) = {

    val t2 = Frame(t.toColSeq.flatMap {
      case (cx, series) =>
        marginallySignificantColumns.get(cx).map(x => cx -> series * x)
    }: _*)

    val s = t2
      .filterIx(cx => cx != "SalePrice")
      .mapVec(_.rank())
      .T
      .sum
      .mapValues(math.log)

    val ranksumFrame =
      Frame("ranksum" -> s, "ranksum_2" -> s * s, "ranksum_3" -> s * s * s)

    ranksumFrame
  }

  def substituteZerosWithMean(f: Frame[String, String, Double]) = Frame(
    f.toColSeq.map {
      case (cx, series) =>
        if (Set("BsmtFinSF1",
                "MasVnrArea",
                "BsmtUnfSF",
                "2ndFlrSF",
                "WoodDeckSF",
                "OpenPorchSF").contains(cx)) {
          val m = series.toVec.filter(_ > 0d).mean
          (cx, series.mapValues(v => if (v == 0d) m else v))
        } else (cx, series)
    }: _*
  )

  def remodeled(f: Frame[String, String, Double]) =
    Frame(
      "remodeled" -> (f.firstCol("YearBuilt") <> f.firstCol("YearRemodAdd"))
        .mapValues(x => if (x) 1.0 else 0d))

  def interaction(v1: String, v2: String) =
    scala.util.Try {

      val xc = Frame(
        "xc" -> trainingNumeric.firstCol(v1) * trainingNumeric.firstCol(v2))

      val lm1 = stat.regression.LinearRegression.linearRegression(
        data = trainingNumeric.col("SalePrice", v1, v2).rconcat(xc),
        yKey = "SalePrice",
        lambda = 0,
        addIntercept = true
      )

      val (p, b) =
        (lm1.covariates("xc")._2.pValue, lm1.covariates("xc")._1.slope)

      if (lm1.covariates.size == 4 && lm1.intercept._1.slope > 0d && lm1.covariates
            .forall(_._2._2.pValue < 1E-5))
        Some(p, b)
      else None

    }.toOption.flatten

  val interactionsToUse = trainingNumeric.colIx.toSeq
    .filterNot(_ == "SalePrice")
    .combinations(2)
    .flatMap { comb =>
      if (continuousVariables.contains(comb(0)) || continuousVariables
            .contains(comb(1)))
        Some((comb(0), comb(1)))
      else None
    // println()
    }
    .toList
    .filter {
      case (v1, v2) =>
        val s = trainingNumeric.firstCol(v1) * trainingNumeric.firstCol(v2)
        s.toVec.countif(_ == 0d) < 1000
    }
  // .filter {
  //   case (v1, v2) =>
  //     interaction(v1, v2).isDefined
  // }
  // .take(0)

  println(interactionsToUse.size)

  def interactionTerms(f: Frame[String, String, Double]) = Frame(
    interactionsToUse.map {
      case (v1, v2) =>
        v1 + "_x_" + v2 -> f.firstCol(v1) * f.firstCol(v2)
    }: _*
  )

  def log10(f: Frame[String, String, Double]) =
    f.mapVec(_.map(x => math.log10(x + 1))).mapColIndex(x => x + "_log")

  // def rbfFeatures(f: Frame[String, String, Double]) = {
  //
  //   val normalized = normalizeWith(
  //     f.filterIx(cx => stdTraining.index.contains(cx)),
  //     stdTraining)
  //
  //   RadialBasisFunctions
  //     .makeFrame(normalized, rbfCenters._1, rbfCenters._2 * 10)
  //     .mapColIndex(x => "rbf_" + x)
  // }

  val extendedTraining =
    // Frame(
    //   (trainingNumeric.col("SalePrice") rconcat rbfFeatures(
    //     trainingNumeric.filterIx(_ != "SalePrice"))).toRowSeq: _*).T
    (substituteZerosWithMean(extend(trainingNumeric)))
      .rconcat(sumCount(trainingNumeric))
      // .rconcat(interactionTerms(trainingNumeric))
      // .rconcat(log10(trainingNumeric))
      // .rconcat(rbfFeatures(trainingNumeric.filterIx(_ != "SalePrice")))
      .rconcat(remodeled(trainingNumeric))

  val extendedTest =
    (substituteZerosWithMean(extend(testNumeric)))
      .rconcat(sumCount(testNumeric))
      // .rconcat(interactionTerms(testNumeric))
      // .rconcat(log10(testNumeric))
      // .rconcat(rbfFeatures(testNumeric))
      .rconcat(remodeled(testNumeric))

  // val stdTraining = extendedTraining.stdev
  // val rbfCenters1 = RadialBasisFunctions.centers(
  //   normalizeWith(extendedTraining, stdTraining).filterIx(_ != "SalePrice"),
  //   trainingNumeric.rowIx.toSeq //.take(200)
  // )
  //
  // val rbfCenters =
  //   rbfCenters1._1.toRowSeq.map(_._2.toVec).zip(rbfCenters1._2.toVec.toSeq)

  println(extendedTraining.colIx.toSeq.toSet &~ extendedTest.colIx.toSeq.toSet)

  // println(rbfFeatures(trainingNumeric).toRowSeq.map(_._2.toSeq))

  // println(rbfFeatures(testNumeric).row("1583").toSeq)

  // interaction("LotArea", "Fireplaces")

  // singlePlot("YearRemodAdd")
  // singlePlot("GarageYrBlt")
  // singlePlot("GrLivArea")
  // singlePlot("TotRmsAbvGrd")
  // singlePlot("Fireplaces")
  // singlePlot("BedroomAbvGr")
  // singlePlot("ranksum")
  // singlePlot("YrSold")

  def singlePlot(variable: String) = {

    val lm1 = stat.regression.LinearRegression.linearRegression(
      data = extendedTraining.col(
        "SalePrice",
        variable,
        (variable + "_2"),
        (variable + "_3")
      ),
      yKey = "SalePrice",
      lambda = 0,
      addIntercept = true
    )

    println(lm1.table)

    val predicted =
      lm1.predictFrame(extendedTraining, intercept = true)

    val joinedWithPredicted =
      extendedTraining.rconcat(Frame("SalePrice_pred" -> predicted))

    pdfToFile(
      new File(variable + ".pdf"),
      xyplot(
        joinedWithPredicted.col(variable, "SalePrice") -> point(color =
                                                                  Color.black,
                                                                labelText =
                                                                  true),
        joinedWithPredicted
          .col(variable, "SalePrice_pred")
          .sortedRowsBy(_.first(variable))
          -> line(color = Color.red)
      )(xlab = variable, axisMargin = 0.1),
      1000)
  }

  // throw new RuntimeException("boo")
  val linearRegressionResultCV = stat.sgd.Cv.fitWithCV(
    extendedTraining,
    "SalePrice",
    false,
    stat.sgd.LinearRegression,
    stat.sgd.ElasticNet(1d, 1d),
    stat.sgd.CoordinateDescentUpdater(true),
    stat.crossvalidation.Split(0.95, rng),
    stat.crossvalidation.KFold(4, rng, 1),
    stat.crossvalidation.HyperParameterSearch(
      HyperParameterSearch.Random2DGenerator(-1, 3d, -1, 3d)(() =>
        rng.nextDouble),
      DoubleRandom(-1, 2d)(() => rng.nextDouble),
      10),
    IdentityFeatureMapFactory, //RbfFeatureMapFactory(rbfCenters),
    bootstrapAggregate = None,
    maxIterations = 1000000,
    minEpochs = 10.0,
    convergedAverage = 1,
    batchSize = None,
    stop = RelativeStopTraining(1E-6),
    rng,
    true
  )

  fileutils.writeToFile(
    new File("pred.model.json"),
    stat.sgd.NamedSgdResult.write(linearRegressionResultCV._2))

  val backandforth: stat.sgd.NamedSgdResult[_, _] = stat.sgd.NamedSgdResult
    .read(stat.sgd.NamedSgdResult.write(linearRegressionResultCV._2))

  println(linearRegressionResultCV._1)
  println(linearRegressionResultCV._2.raw.estimatesV.countif(_ != 0d))
  // val estimates =
  //   Series(linearRegressionResultCV._2.raw.estimatesV,
  //          linearRegressionResultCV._2.names)
  // println(estimates.filter(x => x != 0.0).toSeq.sortBy(_._2).reverse.size)
  // println(estimates.filter(x => x != 0.0).toSeq.sortBy(_._2).reverse)

  val predicted =
    linearRegressionResultCV._2
      .predictFrame(extendedTraining, intercept = true)

  println(predicted)

  //
  // {
  //   val lm1 = stat.regression.LinearRegression.linearRegression(
  //     data =
  //       trainingNumeric.col("SalePrice").rconcat(Frame("pred1" -> predicted)),
  //     yKey = "SalePrice",
  //     lambda = 0,
  //     addIntercept = true
  //   )
  //
  //   val a = lm1.intercept._1.slope
  //
  //   val b = lm1.covariates("pred1")._1.slope
  //
  //   val predicted2 = predicted.mapValues(x => a + b * x)
  //
  //   println(a)
  //   println(b)
  // }
  //
  pdfToFile(new java.io.File("pred.pdf"),
            xyplot(
              predicted
                .join(trainingNumeric.firstCol("SalePrice"), index.InnerJoin)
                .toRowSeq
                .map {
                  case (id, s) =>
                    (s.toVec.raw(0) -> s.toVec.raw(1))
                })(draw1Line = true),
            1000)

  //           val joinedWithPredicted =
  //             extendedTraining
  //               .rconcat(Frame("SalePrice_pred" -> predicted))
  //               .sortedRowsBy(_.get("ranksum").get)
  // pdfToFile(
  //   new java.io.File("pred2.pdf"),
  //   xyplot(joinedWithPredicted.col("ranksum", "SalePrice"),
  //          joinedWithPredicted.col("ranksum", "SalePrice_pred") -> line(
  //            color = Color.red))(),
  //   1000)

  //
  // val topOutliers = joinedWithPredicted.toSeq
  //   .map(x => x._1 -> math.abs(x._2._1 - x._2._2))
  //   .sortBy(_._2)
  //   .reverse
  //   .take(20)
  //   .map(_._1)
  //
  // val scatterCont = sequence(
  //   trainingX.toColSeq.toList
  //     .filter(x => continuousVariables.contains(x._1))
  //     .map {
  //       case (cx, series) =>
  //         val d: Seq[(Double, Double, Double)] =
  //           (series.toSeq zip trainingY.toSeq).map {
  //             case ((id, c), sp) =>
  //               (c, sp, if (topOutliers.contains(id)) 1d else 0d)
  //           }
  //         xyplot(d -> point())(xLabelRotation = -0.5, main = cx)
  //     },
  //   TableLayout(10))
  //
  // pdfToFile(new File("scatter_cont.pdf"), scatterCont, 1000)
  //
  // val scatterCat = sequence(
  //   trainingX.toColSeq.toList
  //     .filterNot(x => continuousVariables.contains(x._1))
  //     .map {
  //       case (cx, series) =>
  //         val d: Seq[(Double, Double, Double)] =
  //           (series.toSeq zip trainingY.toSeq).map {
  //             case ((id, c), sp) =>
  //               (c, sp, if (topOutliers.contains(id)) 1d else 0d)
  //           }
  //         xyplot(d -> point())(xLabelRotation = -0.5, main = cx)
  //     },
  //   TableLayout(10))
  //
  // pdfToFile(new File("scatter_cat.pdf"), scatterCat, 1000)
  //
  // // Test data
  //
  val predictedTest =
    linearRegressionResultCV._2
      .predictFrame(extendedTest, intercept = true) //* b + a

  // println(extendedTest.first("1583").toSeq)
  // println(linearRegressionResultCV._2.scaledEstimatesV.toSeq)
  // println(
  //   Series(("intercept" -> 1d) +: extendedTest.first("1583").toSeq: _*).toVec vv linearRegressionResultCV._2.scaledEstimatesV)

  fileutils.writeToFile(
    "pred.csv",
    "Id,SalePrice\n" + predictedTest.toSeq
      .map(x => x._1 + "," + math.max(1d, (math.pow(10d, x._2) - 1)))
      .mkString("\n"))
}
