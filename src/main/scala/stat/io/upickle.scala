package stat.io

import upickle.{default, json, Js}
import upickle.default.{Writer, Reader}
import org.saddle._
import stat.sgd._

object upicklers {

  implicit def vec2Writer[T: Writer] = upickle.default.Writer[Vec[T]] {
    case t => Js.Arr(t.toSeq.map(x => implicitly[Writer[T]].write(x)): _*)
  }
  implicit def vec2Reader[T: Reader: ST] = Reader[Vec[T]] {
    case Js.Arr(xs @ _ *) =>
      Vec(xs.map(x => implicitly[Reader[T]].read(x)): _*)
  }

  implicit def index2Writer[T: Writer] = upickle.default.Writer[Index[T]] {
    case t => Js.Arr(t.toSeq.map(x => implicitly[Writer[T]].write(x)): _*)
  }
  implicit def index2Reader[T: Reader: ST: ORD] = Reader[Index[T]] {
    case Js.Arr(xs @ _ *) =>
      Index(xs.map(x => implicitly[Reader[T]].read(x)): _*)
  }

  implicit def series2Writer[T: Writer, X: Writer] =
    upickle.default.Writer[Series[X, T]] {
      case t =>
        Js.Obj("index" -> implicitly[Writer[Index[X]]].write(t.index),
               "values" -> implicitly[Writer[Vec[T]]].write(t.toVec))
    }
  implicit def series2Reader[T: Reader: ST, X: Reader: ST: ORD] =
    upickle.default.Reader[Series[X, T]] {
      case Js.Obj(xs @ _ *) =>
        val m = xs.toMap
        Series(
          implicitly[Reader[Vec[T]]].read(m("values")),
          implicitly[Reader[Index[X]]].read(m("index"))
        )
    }

  implicit def obj2Writer[E, P] =
    upickle.default.Writer[ObjectiveFunction[E, P]] {
      case LinearRegression => Js.Str("linear")
      case LogisticRegression => Js.Str("logistic")
      case MultinomialLogisticRegression(c) =>
        Js.Obj("obj" -> Js.Str("multinomial"), "c" -> Js.Num(c))
    }
  implicit def obj2Reader[E, P] = Reader[ObjectiveFunction[E, P]] {
    case Js.Str("linear") =>
      LinearRegression.asInstanceOf[ObjectiveFunction[E, P]]
    case Js.Str("logistic") =>
      LogisticRegression.asInstanceOf[ObjectiveFunction[E, P]]
    case Js.Obj(xs @ _ *) =>
      val m = xs.toMap
      MultinomialLogisticRegression(m("c").num.toInt)
        .asInstanceOf[ObjectiveFunction[E, P]]
  }

}
