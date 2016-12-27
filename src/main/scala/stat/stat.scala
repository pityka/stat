package stat

sealed trait MissingMode
case object DropSample extends MissingMode
case object MeanImpute extends MissingMode
