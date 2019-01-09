package stat.io

import upickle.default
import upickle.default._
import ujson.Js
import org.saddle._
import stat.sgd._

case class Wrapper[X,T](index: Index[X], values: Vec[T])

object upicklers {

  implicit def vec2Writer[T: ReadWriter: ST: ORD] = 
    upickle.default.readwriter[Seq[T]].bimap[Vec[T]](
      _.toSeq, 
      x => Vec(x:_*)
    )

  implicit def index2Writer[T: ReadWriter: ST: ORD] = 
    upickle.default.readwriter[Seq[T]].bimap[Index[T]](
      _.toSeq, 
      Vec(_:_*)
    )

  implicit def series2Writer[T: ReadWriter, X: ReadWriter](
    implicit rw: ReadWriter[Wrapper[X,T]],
              stx: ST[X],ordx:ORD[X],
              stt: ST[T],
              ordt: ORD[T]) =
    upickle.default.readwriter[Wrapper[X,T]].bimap[Series[X,T]](
      series => Wrapper(series.index,series.toVec),
      wrapper => Series(wrapper.values,wrapper.index)
    )

  implicit def obj2Writer[E, P] =
    upickle.default.readwriter[Js.Value].bimap[ObjectiveFunction[E, P]](
      {
        case LinearRegression => Js.Str("linear")
        case LogisticRegression => Js.Str("logistic")
        case MultinomialLogisticRegression(c) =>
          Js.Obj("obj" -> Js.Str("multinomial"), "c" -> Js.Num(c))
      },
      {
        case Js.Str("linear") =>
          LinearRegression.asInstanceOf[ObjectiveFunction[E, P]]
        case Js.Str("logistic") =>
          LogisticRegression.asInstanceOf[ObjectiveFunction[E, P]]
        case Js.Obj(xs ) =>
          val m = xs.toMap
          MultinomialLogisticRegression(m("c").num.toInt)
            .asInstanceOf[ObjectiveFunction[E, P]]
        case _ => ???
      }
    ) 

}
