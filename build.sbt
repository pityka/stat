scalaVersion := "2.11.8"

name := "stat"

organization := "io.github.pityka"

version := "0.0.8"

libraryDependencies ++= Seq(
  "io.github.pityka" %% "saddle-linalg" % "0.0.14",
  "io.github.pityka" % "hierarchical-clustering-fork" % "1.0-5",
  "io.github.pityka" %% "nspl-saddle" % "0.0.12",
  "io.github.pityka" %% "nspl-awt" % "0.0.12" % "test",
  "org.scalatest" %% "scalatest" % "3.0.0" % "test",
  "biz.enef" %% "slogging" % "0.5.1"
)

reformatOnCompileSettings

parallelExecution in Test := false

pomExtra in Global := {
  <url>https://pityka.github.io/stat</url>
  <licenses>
    <license>
      <name>MIT</name>
      <url>https://opensource.org/licenses/MIT</url>
    </license>
  </licenses>
  <scm>
    <connection>scm:git:github.com/pityka/stat</connection>
    <developerConnection>scm:git:git@github.com:pityka/stat</developerConnection>
    <url>github.com/pityka/stat</url>
  </scm>
  <developers>
    <developer>
      <id>pityka</id>
      <name>Istvan Bartha</name>
      <url>https://pityka.github.io/stat/</url>
    </developer>
  </developers>
}
