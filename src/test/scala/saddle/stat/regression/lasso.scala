package stat.regression

import org.saddle._
import org.scalatest._
import stat._
import org.saddle.io._

class LassoSuite extends FunSpec with Matchers {
  describe("lasso alone") {

    val frame = CsvParser
      .parse(CsvFile(getClass.getResource("/").getPath + "/example.csv"))
      .withColIndex(0)
      .withRowIndex(0)
      .mapValues(_.toDouble)

    it("lambda=0") {

      val lassoresult = LASSO.fit(frame, "V22", 0.0, addIntercept = true)

      val ridgeresult = Ridge.fit(frame, "V22", 0.0, addIntercept = true)

      val sgdresult =
        stat.sgd.Sgd.optimize(frame,
                              "V22",
                              stat.sgd.LinearRegression,
                              stat.sgd.L2(0d),
                              stat.sgd.NewtonUpdater)

      val ln =
        LinearRegression.linearRegression(frame, "V22", addIntercept = true)

      val glmnet = Vector(
        0.110947733,
        1.378783189,
        0.023275142,
        0.763496730,
        0.060802214,
        -0.901598952,
        0.614425728,
        0.118277706,
        0.397670502,
        -0.031395521,
        0.128445909,
        0.247276619,
        -0.064775290,
        -0.045980087,
        -1.159522599,
        -0.138015549,
        -0.045678862,
        -0.048419881,
        0.052383392,
        -0.002729424,
        -1.144092052
      )

      ((ln.estimatesV.toSeq) zip lassoresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          (math.abs(x - y) < 0.01) should equal(true)
      }

      (glmnet zip ridgeresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          (math.abs(x - y) < 0.01) should equal(true)
      }

      (glmnet zip lassoresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          (math.abs(x - y) < 0.01) should equal(true)
      }

      (glmnet zip sgdresult.estimates.toSeq).foreach {
        case (x, y) =>
          (math.abs(x - y) < 0.01) should equal(true)
      }

    }

    it("lambda=0.5") {

      val lassoresult = LASSO.fit(frame, "V22", 0.5)

      // val sgdresult2 =
      //   stat.sgd.Sgd.optimize(frame,
      //                         "V22",
      //                         stat.sgd.L2(stat.sgd.LinearRegression, 50),
      //                         standardize = false)

      // println("XXX")
      // println(sgdresult2)
      //
      // val sgdresult =
      //   stat.sgd.Sgd.optimize(frame,
      //                         "V22",
      //                         stat.sgd.L2Prox(stat.sgd.LinearRegression, 50),
      //                         standardize = false)
      //
      // println(sgdresult)

      // this is glmnet lambda=0.5/200 because they use different weighting inside the objective function
      val glmnet = Vector(
        0.110418116,
        1.379439342,
        0.023714461,
        0.764764752,
        0.062580731,
        -0.903005714,
        0.615777699,
        0.120207005,
        0.398766538,
        -0.033049035,
        0.130978034,
        0.248744614,
        -0.066345896,
        -0.047107247,
        -1.160905703,
        -0.140931254,
        -0.047584249,
        -0.050790214,
        0.053901783,
        -0.003888988,
        -1.145513852
      )

      val penalized = Vector(
        0.111620533,
        1.377918731,
        0.022577490,
        0.761963299,
        0.058647387,
        -0.899923091,
        0.613034669,
        0.116095232,
        0.396443319,
        -0.029603638,
        0.125616865,
        0.245748790,
        -0.062984463,
        -0.044669540,
        -1.157973887,
        -0.134747133,
        -0.043526123,
        -0.045685510,
        0.050687485,
        -0.001390277,
        -1.142443919
      )

      (penalized zip lassoresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          if (!(math.abs(x - y) < 0.01)) {
            println("penalized: " + x + " vs my: " + y)
          }
          (math.abs(x - y) < 0.01) should equal(true)
      }

      (glmnet zip lassoresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          if (!(math.abs(x - y) < 0.01)) {
            println("glmnet: " + x + " vs my: " + y)
          }
          (math.abs(x - y) < 0.01) should equal(true)
      }

      // (glmnet zip sgdresult.estimates.toSeq).foreach {
      //   case (x, y) =>
      //     (math.abs(x - y) < 0.01) should equal(true)
      // }

    }

    it("lambda=50 against penalized") {
      val lassoresult = LASSO.fit(frame, "V22", 50.0)

      val penalized = Vector(
        0.25867058,
        1.03666762,
        0.00000000,
        0.26086084,
        0.00000000,
        -0.38564371,
        0.26790191,
        0.00000000,
        0.01054177,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        -0.86703516,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        0.00000000,
        -0.46707087
      )

      (penalized zip lassoresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          if (!(math.abs(x - y) < 0.01)) {
            println("penalized: " + x + " vs my: " + y)
          }
          (math.abs(x - y) < 0.02) should equal(true)
      }

      // (glmnet zip lassoresult.coefficients.toSeq).foreach {
      //   case (x, y) =>
      //     if (!(math.abs(x - y) < 0.01)) {
      //       println("glmnet: " + x + " vs my: " + y)
      //     }
      //     (math.abs(x - y) < 0.01) should equal(true)
      // }

    }

    it("lambda=50 against penalized L2") {

      val result = Ridge.fit(frame, "V22", 50d)

      val sgdresult =
        stat.sgd.Sgd.optimize(frame,
                              "V22",
                              stat.sgd.LinearRegression,
                              stat.sgd.L2(50d),
                              stat.sgd.NewtonUpdater,
                              standardize = false)

      val penalized = IndexedSeq(0.305767252,
                                 0.973779894,
                                 0.068934404,
                                 0.497185503,
                                 -0.043501453,
                                 -0.603505652,
                                 0.517660241,
                                 0.112092970,
                                 0.282919550,
                                 -0.058310925,
                                 0.010746792,
                                 0.173727036,
                                 -0.057283446,
                                 -0.006831348,
                                 -0.826476739,
                                 -0.030049603,
                                 0.013115346,
                                 0.017033835,
                                 0.061459813,
                                 0.014791350,
                                 -0.686263503)

      (penalized zip result.estimatesV.toSeq).foreach {
        case (x, y) =>
          if (!(math.abs(x - y) < 0.01)) {
            // println("penalized: " + x + " vs my: " + y)
          }
          (math.abs(x - y) < 0.1) should equal(true)
      }

      (penalized zip sgdresult.estimates.toSeq).foreach {
        case (x, y) =>
          if (!(math.abs(x - y) < 0.01)) {
            println("penalized: " + x + " vs my: " + y)
          }
          (math.abs(x - y) < 0.01) should equal(true)
      }

    }

    it("scad lambda=50 against ncvreg") {
      val lassoresult =
        PenalizedWithSCAD.fit(frame, "V22", 10)

      val ncvreg = Seq(
        0.1308801426,
        1.3796539851,
        0.0000000000,
        0.7659849860,
        0.0072198822,
        -0.9020176288,
        0.6060924633,
        0.0447846110,
        0.3972901012,
        0.0000000000,
        0.0472768243,
        0.2569556151,
        0.0000000000,
        -0.0092788275,
        -1.1364997738,
        -0.0331382513,
        -0.0005243524,
        0.0000000000,
        0.0003476129,
        0.0000000000,
        -1.1584229366
      )

      // println(lassoresult.coefficients)

      (ncvreg zip lassoresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          if (!(math.abs(x - y) < 0.2)) {
            println("penalized: " + x + " vs my: " + y)
          }
          (math.abs(x - y) < 0.2) should equal(true)
      }

      // (glmnet zip lassoresult.coefficients.toSeq).foreach {
      //   case (x, y) =>
      //     if (!(math.abs(x - y) < 0.01)) {
      //       println("glmnet: " + x + " vs my: " + y)
      //     }
      //     (math.abs(x - y) < 0.01) should equal(true)
      // }

    }

    it("elastic net  against penalized") {

      val lassoresult = PenalizedWithElasticNet.fit(frame, "V22", 50d -> 25d)

      val penalized = Seq(
        0.329945491,
        0.849655715,
        0.000000000,
        0.206803964,
        0.000000000,
        -0.322154692,
        0.261500178,
        0.000000000,
        0.008798273,
        0.000000000,
        0.000000000,
        0.000000000,
        0.000000000,
        0.000000000,
        -0.723725582,
        0.000000000,
        0.000000000,
        0.000000000,
        0.000000000,
        0.000000000,
        -0.363951587
      )

      // println(lassoresult.coefficients)

      (penalized zip lassoresult.estimatesV.toSeq).foreach {
        case (x, y) =>
          if (!(math.abs(x - y) < 0.1)) {
            println("penalized: " + x + " vs my: " + y)
          }
          (math.abs(x - y) < 0.1) should equal(true)
      }

      // (glmnet zip lassoresult.coefficients.toSeq).foreach {
      //   case (x, y) =>
      //     if (!(math.abs(x - y) < 0.01)) {
      //       println("glmnet: " + x + " vs my: " + y)
      //     }
      //     (math.abs(x - y) < 0.01) should equal(true)
      // }

    }

  }
}
